//! Vulkan implementations of the candle-nn custom ops.
//!
//! Each `native_*` helper returns `Some(storage)` when the op ran as a Vulkan
//! compute kernel. It returns `None` when there is no native kernel for the
//! case (non-f32 dtype, non-contiguous input, kernels disabled) or the launch
//! failed. Every `None` is recorded in the Vulkan fallback counter, and the
//! caller then runs the CPU round-trip ([`round_trip1`] and friends), so CPU
//! execution is never reported as GPU execution.
use candle::backend::{BackendDevice, BackendStorage};
use candle::vulkan_backend::kernels;
use candle::{CpuStorage, DType, Layout, Result, Shape, VulkanStorage};

fn note(op: &str, err: Option<&candle::Error>) {
    kernels::note_fallback(op, err)
}

fn finish(op: &str, r: Result<()>, out: VulkanStorage) -> Option<VulkanStorage> {
    match r {
        Ok(()) => Some(out),
        Err(e) => {
            note(op, Some(&e));
            None
        }
    }
}

pub(crate) fn native_sigmoid(s: &VulkanStorage, l: &Layout) -> Result<Option<VulkanStorage>> {
    if s.dtype() != DType::F32 || !kernels::native_enabled() {
        note("sigmoid", None);
        return Ok(None);
    }
    let code = kernels::unary_code("sigmoid").expect("sigmoid has a unary kernel");
    let n = l.shape().elem_count();
    let out = s.device.alloc(DType::F32, n)?;
    let r = kernels::run_unary(&s.device, code, s.buf(), out.buf(), n, l);
    Ok(finish("sigmoid", r, out))
}

pub(crate) fn native_softmax_last_dim(
    s: &VulkanStorage,
    l: &Layout,
) -> Result<Option<VulkanStorage>> {
    let op = "softmax-last-dim";
    let cols = l.dims().last().copied().unwrap_or(0);
    let offs = l.contiguous_offsets();
    if s.dtype() != DType::F32 || !kernels::native_enabled() || cols == 0 || offs.is_none() {
        note(op, None);
        return Ok(None);
    }
    let (o1, _) = offs.unwrap();
    let rows = l.shape().elem_count() / cols;
    let out = s.device.alloc(DType::F32, rows * cols)?;
    let r = kernels::run_softmax_last(&s.device, s.buf(), out.buf(), rows, cols, o1);
    Ok(finish(op, r, out))
}

pub(crate) fn native_rms_norm(
    s1: &VulkanStorage,
    l1: &Layout,
    s2: &VulkanStorage,
    l2: &Layout,
    eps: f32,
) -> Result<Option<VulkanStorage>> {
    let op = "rms-norm";
    let cols = l1.dims().last().copied().unwrap_or(0);
    let eligible = s1.dtype() == DType::F32
        && s2.dtype() == DType::F32
        && kernels::native_enabled()
        && cols > 0
        && l2.shape().elem_count() == cols;
    match (eligible, l1.contiguous_offsets(), l2.contiguous_offsets()) {
        (true, Some((o1, _)), Some((a1, _))) => {
            let rows = l1.shape().elem_count() / cols;
            let out = s1.device.alloc(DType::F32, rows * cols)?;
            let r = kernels::run_rmsnorm(
                &s1.device,
                s1.buf(),
                s2.buf(),
                out.buf(),
                rows,
                cols,
                o1,
                a1,
                eps,
            );
            Ok(finish(op, r, out))
        }
        _ => {
            note(op, None);
            Ok(None)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn native_rope(
    op: &str,
    interleaved: bool,
    s1: &VulkanStorage,
    l1: &Layout,
    s2: &VulkanStorage,
    l2: &Layout,
    s3: &VulkanStorage,
    l3: &Layout,
) -> Result<Option<VulkanStorage>> {
    let eligible = s1.dtype() == DType::F32
        && s2.dtype() == DType::F32
        && s3.dtype() == DType::F32
        && kernels::native_enabled();
    match (
        eligible,
        l1.contiguous_offsets(),
        l2.contiguous_offsets(),
        l3.contiguous_offsets(),
        l1.shape().dims4(),
    ) {
        (true, Some((o1, _)), Some((c1, _)), Some((n1, _)), Ok((b, h, t, d))) => {
            let cs_batched = l2.dims().len() == 3 && l3.dims().len() == 3;
            let out = s1.device.alloc(DType::F32, b * h * t * d)?;
            let r = kernels::run_rope(
                &s1.device,
                interleaved,
                s1.buf(),
                s2.buf(),
                s3.buf(),
                out.buf(),
                (b, h, t, d),
                o1,
                c1,
                n1,
                cs_batched,
            );
            Ok(finish(op, r, out))
        }
        _ => {
            note(op, None);
            Ok(None)
        }
    }
}

/// CPU round-trip for a unary op (the fallback has already been recorded).
pub(crate) fn round_trip1(
    s: &VulkanStorage,
    l: &Layout,
    f: impl FnOnce(&CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
) -> Result<(VulkanStorage, Shape)> {
    let c = s.to_cpu_storage()?;
    let (out, shape) = f(&c, l)?;
    Ok((s.device().storage_from_cpu_storage(&out)?, shape))
}

/// CPU round-trip for a binary op (the fallback has already been recorded).
pub(crate) fn round_trip2(
    s1: &VulkanStorage,
    l1: &Layout,
    s2: &VulkanStorage,
    l2: &Layout,
    f: impl FnOnce(&CpuStorage, &Layout, &CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
) -> Result<(VulkanStorage, Shape)> {
    let c1 = s1.to_cpu_storage()?;
    let c2 = s2.to_cpu_storage()?;
    let (out, shape) = f(&c1, l1, &c2, l2)?;
    Ok((s1.device().storage_from_cpu_storage(&out)?, shape))
}

/// CPU round-trip for a ternary op (the fallback has already been recorded).
#[allow(clippy::type_complexity)]
pub(crate) fn round_trip3(
    s1: &VulkanStorage,
    l1: &Layout,
    s2: &VulkanStorage,
    l2: &Layout,
    s3: &VulkanStorage,
    l3: &Layout,
    f: impl FnOnce(
        &CpuStorage,
        &Layout,
        &CpuStorage,
        &Layout,
        &CpuStorage,
        &Layout,
    ) -> Result<(CpuStorage, Shape)>,
) -> Result<(VulkanStorage, Shape)> {
    let c1 = s1.to_cpu_storage()?;
    let c2 = s2.to_cpu_storage()?;
    let c3 = s3.to_cpu_storage()?;
    let (out, shape) = f(&c1, l1, &c2, l2, &c3, l3)?;
    Ok((s1.device().storage_from_cpu_storage(&out)?, shape))
}
