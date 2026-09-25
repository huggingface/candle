//! Native SYCL kernels for the candle-nn fused ops (softmax / rms_norm / rope).
//! Entry points called from `candle-nn`'s `sycl_fwd` impls, replacing the
//! host round-trip shim.
use super::{ffi_layout, k, storage_from_buffer, to_sycl_dtype, wrap, SyclStorage};
use crate::backend::BackendStorage;
use crate::{Layout, Result};

fn same_dense(s: &SyclStorage, numel: usize) -> Result<SyclStorage> {
    let buf = wrap(k::DeviceBuffer::alloc(
        s.device().queue(),
        numel * s.dtype().size_in_bytes(),
    ))?;
    Ok(storage_from_buffer(s.device(), buf, s.dtype(), numel))
}

/// A contiguous, zero-offset view of a storage: either the original, or a fresh
/// dense copy (for offset views such as a KV-cache slice). The kernels below all
/// assume dense buffers. `SyclStorage` is not `Clone`, hence the hand-rolled Cow.
enum Dense<'a> {
    Borrowed(&'a SyclStorage),
    Owned(SyclStorage),
}
impl Dense<'_> {
    fn get(&self) -> &SyclStorage {
        match self {
            Dense::Borrowed(s) => s,
            Dense::Owned(s) => s,
        }
    }
}

fn densify<'a>(s: &'a SyclStorage, l: &Layout, what: &str) -> Result<Dense<'a>> {
    if l.start_offset() == 0 && l.is_contiguous() {
        return Ok(Dense::Borrowed(s));
    }
    if !l.is_contiguous() {
        crate::bail!("sycl {what}: input must be contiguous");
    }
    let numel = l.shape().elem_count();
    let dense = same_dense(s, numel)?;
    wrap(k::copy_strided(
        s.device().queue(),
        to_sycl_dtype(s.dtype())?,
        &ffi_layout(l)?,
        s.buf(),
        dense.buf(),
        0,
        numel,
    ))?;
    Ok(Dense::Owned(dense))
}

pub fn softmax_last_dim(s: &SyclStorage, l: &Layout) -> Result<SyclStorage> {
    let s = densify(s, l, "softmax")?;
    let s = s.get();
    let dims = l.dims();
    let d = dims[dims.len() - 1];
    let rows = l.shape().elem_count() / d.max(1);
    let out = same_dense(s, l.shape().elem_count())?;
    wrap(k::softmax_lastdim(
        s.device().queue(),
        to_sycl_dtype(s.dtype())?,
        s.buf(),
        out.buf(),
        rows,
        d,
    ))?;
    Ok(out)
}

/// Elementwise sigmoid `1 / (1 + e^-x)`. candle-nn's `Sigmoid` is a `CustomOp1`
/// with no candle-core builtin, so it routes here rather than through
/// `unary_impl`.
pub fn sigmoid(s: &SyclStorage, l: &Layout) -> Result<SyclStorage> {
    let s = densify(s, l, "sigmoid")?;
    let s = s.get();
    let n = l.shape().elem_count();
    let out = same_dense(s, n)?;
    wrap(k::unary(
        s.device().queue(),
        k::UnaryOp::Sigmoid,
        to_sycl_dtype(s.dtype())?,
        &k::Layout::dense(),
        s.buf(),
        out.buf(),
        n,
    ))?;
    Ok(out)
}

pub fn rms_norm(
    x: &SyclStorage,
    xl: &Layout,
    alpha: &SyclStorage,
    al: &Layout,
    eps: f32,
) -> Result<SyclStorage> {
    let x = densify(x, xl, "rms_norm x")?;
    let x = x.get();
    let alpha = densify(alpha, al, "rms_norm alpha")?;
    let alpha = alpha.get();
    let dims = xl.dims();
    let d = dims[dims.len() - 1];
    let rows = xl.shape().elem_count() / d.max(1);
    let out = same_dense(x, xl.shape().elem_count())?;
    wrap(k::rms_norm(
        x.device().queue(),
        to_sycl_dtype(x.dtype())?,
        x.buf(),
        alpha.buf(),
        out.buf(),
        rows,
        d,
        eps,
    ))?;
    Ok(out)
}

/// `mode`: 0 interleaved (b,h,t,d), 1 half-split (b,h,t,d), 2 half-split (b,t,h,d).
#[allow(clippy::too_many_arguments)]
pub fn rope(
    mode: u32,
    x: &SyclStorage,
    xl: &Layout,
    cos: &SyclStorage,
    cl: &Layout,
    sin: &SyclStorage,
    sl: &Layout,
) -> Result<SyclStorage> {
    let x = densify(x, xl, "rope x")?;
    let x = x.get();
    let cos = densify(cos, cl, "rope cos")?;
    let cos = cos.get();
    let sin = densify(sin, sl, "rope sin")?;
    let sin = sin.get();
    let (d0, d1, d2, d3) = xl.shape().dims4()?;
    // mode 0/1: (b, h, t, d); mode 2: (b, t, h, d)
    let (b, h, t, d) = if mode == 2 {
        (d0, d2, d1, d3)
    } else {
        (d0, d1, d2, d3)
    };
    let cos_batched = cl.dims().len() == 3;
    let out = same_dense(x, xl.shape().elem_count())?;
    wrap(k::rope(
        x.device().queue(),
        mode,
        to_sycl_dtype(x.dtype())?,
        x.buf(),
        cos.buf(),
        sin.buf(),
        out.buf(),
        b,
        h,
        t,
        d,
        cos_batched,
    ))?;
    Ok(out)
}
