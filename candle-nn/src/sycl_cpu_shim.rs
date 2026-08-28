//! Phase 5 fallback for candle-nn fused ops on the SYCL backend.
//!
//! These `CustomOp` impls have no CPU fallback in candle: on a SYCL tensor the
//! op errors. Until each has a native kernel, `sycl_fwd` copies to the host,
//! runs the existing `cpu_fwd`, and copies the result back. Correct, just an
//! extra host round-trip per call (small tensors — a norm / rope / softmax on
//! one token). Native kernels for the hot ones (rmsnorm, rope, softmax) come
//! next; see `todo.md`.
#![cfg(feature = "sycl")]
use candle::backend::{BackendDevice, BackendStorage};
use candle::{CpuStorage, Layout, Result, Shape, SyclStorage};

fn back(s: &SyclStorage, out: CpuStorage, shape: Shape) -> Result<(SyclStorage, Shape)> {
    Ok((s.device().storage_from_cpu_storage(&out)?, shape))
}

pub fn via_cpu_1(
    s: &SyclStorage,
    l: &Layout,
    f: impl FnOnce(&CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    let (out, shape) = f(&s.to_cpu_storage()?, l)?;
    back(s, out, shape)
}

#[allow(clippy::too_many_arguments)]
pub fn via_cpu_2(
    s1: &SyclStorage,
    l1: &Layout,
    s2: &SyclStorage,
    l2: &Layout,
    f: impl FnOnce(&CpuStorage, &Layout, &CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    let (out, shape) = f(&s1.to_cpu_storage()?, l1, &s2.to_cpu_storage()?, l2)?;
    back(s1, out, shape)
}

#[allow(clippy::too_many_arguments)]
pub fn via_cpu_3(
    s1: &SyclStorage,
    l1: &Layout,
    s2: &SyclStorage,
    l2: &Layout,
    s3: &SyclStorage,
    l3: &Layout,
    f: impl FnOnce(
        &CpuStorage,
        &Layout,
        &CpuStorage,
        &Layout,
        &CpuStorage,
        &Layout,
    ) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    let (out, shape) = f(
        &s1.to_cpu_storage()?,
        l1,
        &s2.to_cpu_storage()?,
        l2,
        &s3.to_cpu_storage()?,
        l3,
    )?;
    back(s1, out, shape)
}

// --- native kernels (fall back to the CPU shim when the fast path can't apply) ---

fn float_contig(s: &SyclStorage, l: &Layout) -> bool {
    use candle::backend::BackendStorage;
    use candle::DType::{BF16, F16, F32};
    // `nn_ops` densifies an offset view itself; only true non-contiguity falls back.
    l.is_contiguous() && matches!(s.dtype(), F16 | BF16 | F32)
}

pub fn softmax_last_dim(
    s: &SyclStorage,
    l: &Layout,
    cpu: impl FnOnce(&CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    if float_contig(s, l) {
        let out = candle::sycl_backend::nn_ops::softmax_last_dim(s, l)?;
        return Ok((out, l.shape().clone()));
    }
    via_cpu_1(s, l, cpu)
}

pub fn sigmoid(
    s: &SyclStorage,
    l: &Layout,
    cpu: impl FnOnce(&CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    if float_contig(s, l) {
        let out = candle::sycl_backend::nn_ops::sigmoid(s, l)?;
        return Ok((out, l.shape().clone()));
    }
    via_cpu_1(s, l, cpu)
}

pub fn rms_norm(
    eps: f32,
    s1: &SyclStorage,
    l1: &Layout,
    s2: &SyclStorage,
    l2: &Layout,
    cpu: impl FnOnce(&CpuStorage, &Layout, &CpuStorage, &Layout) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    if float_contig(s1, l1) && float_contig(s2, l2) {
        let out = candle::sycl_backend::nn_ops::rms_norm(s1, l1, s2, l2, eps)?;
        return Ok((out, l1.shape().clone()));
    }
    via_cpu_2(s1, l1, s2, l2, cpu)
}

/// `mode`: 0 interleaved (b,h,t,d), 1 half-split (b,h,t,d), 2 half-split (b,t,h,d).
#[allow(clippy::too_many_arguments)]
pub fn rope(
    mode: u32,
    s1: &SyclStorage,
    l1: &Layout,
    s2: &SyclStorage,
    l2: &Layout,
    s3: &SyclStorage,
    l3: &Layout,
    cpu: impl FnOnce(
        &CpuStorage,
        &Layout,
        &CpuStorage,
        &Layout,
        &CpuStorage,
        &Layout,
    ) -> Result<(CpuStorage, Shape)>,
) -> Result<(SyclStorage, Shape)> {
    if l1.shape().dims().len() == 4
        && float_contig(s1, l1)
        && float_contig(s2, l2)
        && float_contig(s3, l3)
    {
        let out = candle::sycl_backend::nn_ops::rope(mode, s1, l1, s2, l2, s3, l3)?;
        return Ok((out, l1.shape().clone()));
    }
    via_cpu_3(s1, l1, s2, l2, s3, l3, cpu)
}
