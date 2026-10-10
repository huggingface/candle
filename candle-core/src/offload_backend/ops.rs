//! GPU paths. Every function returns `Ok(None)` when the dtype or layout is
//! outside what the kernels handle, and the caller then routes the op through
//! the CPU backend. That keeps the backend total -- llama runs end to end --
//! while making the GPU coverage explicit and measurable rather than implicit.
use super::{ffi, gpu_allowed, OffloadStorage};
use crate::{DType, Layout, Result, Shape};
use super::kernels as k;

/// Map candle's `UnaryOpT::NAME` onto a kernel opcode. The ops missing here are
/// the ones no_std cannot express: `erf` and `gelu_erf` need `erf`, and
/// `ceil`/`floor`/`round` have no reachable intrinsic (G12).
pub(crate) fn unary_code(name: &str) -> Option<usize> {
    Some(match name {
        "neg" => k::U_NEG,
        "sqr" => k::U_SQR,
        "sqrt" => k::U_SQRT,
        "exp" => k::U_EXP,
        "log" => k::U_LOG,
        "sin" => k::U_SIN,
        "cos" => k::U_COS,
        "abs" => k::U_ABS,
        "recip" => k::U_RECIP,
        "relu" => k::U_RELU,
        "silu" => k::U_SILU,
        "gelu" => k::U_GELU,
        "tanh" => k::U_TANH,
        "sign" => k::U_SIGN,
        _ => return None,
    })
}

pub(crate) fn binary_code(name: &str) -> Option<usize> {
    Some(match name {
        "add" => k::OP_ADD,
        "sub" => k::OP_SUB,
        "mul" => k::OP_MUL,
        "div" => k::OP_DIV,
        "maximum" => k::OP_MAX,
        "minimum" => k::OP_MIN,
        _ => return None,
    })
}

pub(crate) fn unary(s: &OffloadStorage, l: &Layout, name: &str) -> Result<Option<OffloadStorage>> {
    if !gpu_allowed("unary") || s.dtype != DType::F32 || !l.is_contiguous() {
        return Ok(None);
    }
    let Some(code) = unary_code(name) else {
        return Ok(None);
    };
    let n = l.shape().elem_count();
    let out = s.device.alloc_elems(n, DType::F32)?;
    unsafe { k::candle_offload_unary_f32(code, s.at(l.start_offset()), out.ptr(), n) };
    ffi::note_launch();
    ffi::note_gpu_op();
    Ok(Some(out))
}

pub(crate) fn binary(
    lhs: &OffloadStorage,
    rhs: &OffloadStorage,
    lhs_l: &Layout,
    rhs_l: &Layout,
    name: &str,
) -> Result<Option<OffloadStorage>> {
    if !gpu_allowed("binary") || lhs.dtype != DType::F32 || rhs.dtype != DType::F32 {
        return Ok(None);
    }
    // Broadcast layouts (stride 0) are not contiguous, so they land on the CPU
    // path. Handling them on the GPU needs per-dimension strides in the kernel.
    if !lhs_l.is_contiguous() || !rhs_l.is_contiguous() {
        return Ok(None);
    }
    let n = lhs_l.shape().elem_count();
    if n != rhs_l.shape().elem_count() {
        return Ok(None);
    }
    let Some(code) = binary_code(name) else {
        return Ok(None);
    };
    let out = lhs.device.alloc_elems(n, DType::F32)?;
    unsafe { k::candle_offload_binary_f32(
        code,
        lhs.at(lhs_l.start_offset()),
        rhs.at(rhs_l.start_offset()),
        out.ptr(),
        n,
    ) };
    ffi::note_launch();
    ffi::note_gpu_op();
    Ok(Some(out))
}

pub(crate) fn affine(
    s: &OffloadStorage,
    l: &Layout,
    mul: f64,
    add: f64,
) -> Result<Option<OffloadStorage>> {
    if !gpu_allowed("affine") || s.dtype != DType::F32 || !l.is_contiguous() {
        return Ok(None);
    }
    let n = l.shape().elem_count();
    let out = s.device.alloc_elems(n, DType::F32)?;
    unsafe { k::candle_offload_affine_f32(s.at(l.start_offset()), out.ptr(), n, mul as f32, add as f32) };
    ffi::note_launch();
    ffi::note_gpu_op();
    Ok(Some(out))
}

pub(crate) fn const_set(
    s: &mut OffloadStorage,
    v: crate::scalar::Scalar,
    l: &Layout,
) -> Result<bool> {
    if !gpu_allowed("const_set") || s.dtype != DType::F32 || !l.is_contiguous() {
        return Ok(false);
    }
    let n = l.shape().elem_count();
    unsafe { k::candle_offload_fill_f32(s.at(l.start_offset()), n, v.to_f64() as f32) };
    ffi::note_launch();
    ffi::note_gpu_op();
    Ok(true)
}

/// Batched GEMM.
///
/// candle hands over `(batch, m, n, k)` plus both layouts, and the right-hand
/// side is usually a *view*: `Linear` computes `x.matmul(&w.t())`, so `rhs_l`
/// has the transposed strides of a contiguous `(n, k)` weight. Passing those
/// strides to the kernel avoids materialising the transpose, which would
/// otherwise cost a full extra pass over the weights on every single layer.
pub(crate) fn matmul(
    lhs: &OffloadStorage,
    rhs: &OffloadStorage,
    (b, m, n, kk): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<Option<OffloadStorage>> {
    if !gpu_allowed("matmul") || lhs.dtype != DType::F32 || rhs.dtype != DType::F32 {
        return Ok(None);
    }
    let (ls, rs) = (lhs_l.stride(), rhs_l.stride());
    let rank = ls.len();
    if rank < 2 || rs.len() != rank {
        return Ok(None);
    }
    // The kernel walks A as row-major within a batch.
    if ls[rank - 2] != kk || ls[rank - 1] != 1 {
        return Ok(None);
    }
    let (brs, bcs) = (rs[rank - 2], rs[rank - 1]);

    let (a_bs, b_bs) = if b == 1 {
        (0, 0)
    } else {
        if rank < 3 {
            return Ok(None);
        }
        // Only a single effective batch dimension is handled; anything wider
        // would need the leading dims to be provably contiguous.
        if lhs_l.dims()[..rank - 3].iter().any(|&d| d != 1)
            || rhs_l.dims()[..rank - 3].iter().any(|&d| d != 1)
        {
            return Ok(None);
        }
        (ls[rank - 3], rs[rank - 3])
    };

    let out = lhs.device.alloc_elems(b * m * n, DType::F32)?;
    unsafe { k::candle_offload_gemm_f32(
        lhs.at(lhs_l.start_offset()),
        rhs.at(rhs_l.start_offset()),
        out.ptr(),
        b,
        m,
        n,
        kk,
        brs,
        bcs,
        a_bs,
        b_bs,
    ) };
    ffi::note_launch();
    ffi::note_gpu_op();
    Ok(Some(out))
}

// ---------------------------------------------------------------------------
// The ops candle-nn models as CustomOps
// ---------------------------------------------------------------------------
//
// Each returns `Ok(None)` for shapes it does not handle, and candle-nn then
// falls back to the op's own CPU kernel. These three matter more than their
// arithmetic suggests: in a transformer they run once or twice per layer, so
// leaving them on the CPU means three host round trips per layer per token.

/// `dst = src / sqrt(mean(row^2) + eps) * alpha` over the innermost dimension.
pub fn rms_norm(
    s1: &OffloadStorage,
    l1: &Layout,
    s2: &OffloadStorage,
    l2: &Layout,
    eps: f32,
) -> Result<Option<(OffloadStorage, Shape)>> {
    if !gpu_allowed("rms_norm") || s1.dtype != DType::F32 || s2.dtype != DType::F32 {
        return Ok(None);
    }
    if !l1.is_contiguous() || !l2.is_contiguous() {
        return Ok(None);
    }
    let dims = l1.shape().dims();
    let Some(&cols) = dims.last() else {
        return Ok(None);
    };
    if cols == 0 || l2.shape().elem_count() != cols {
        return Ok(None);
    }
    let n = l1.shape().elem_count();
    let out = s1.device.alloc_elems(n, DType::F32)?;
    unsafe {
        k::candle_offload_rms_norm_f32(
            s1.at(l1.start_offset()),
            s2.at(l2.start_offset()),
            out.ptr(),
            n / cols,
            cols,
            eps,
        )
    };
    ffi::note_launch();
    ffi::note_gpu_op();
    Ok(Some((out, l1.shape().clone())))
}

pub fn softmax_last_dim(
    s: &OffloadStorage,
    l: &Layout,
) -> Result<Option<(OffloadStorage, Shape)>> {
    if !gpu_allowed("softmax") || s.dtype != DType::F32 || !l.is_contiguous() {
        return Ok(None);
    }
    let dims = l.shape().dims();
    let Some(&cols) = dims.last() else {
        return Ok(None);
    };
    if cols == 0 {
        return Ok(None);
    }
    let n = l.shape().elem_count();
    let out = s.device.alloc_elems(n, DType::F32)?;
    unsafe { k::candle_offload_softmax_last_f32(s.at(l.start_offset()), out.ptr(), n / cols, cols) };
    ffi::note_launch();
    ffi::note_gpu_op();
    Ok(Some((out, l.shape().clone())))
}

/// Contiguous rotary embedding over a (b, h, t, d) source.
pub fn rope(
    s1: &OffloadStorage,
    l1: &Layout,
    s2: &OffloadStorage,
    l2: &Layout,
    s3: &OffloadStorage,
    l3: &Layout,
) -> Result<Option<(OffloadStorage, Shape)>> {
    if !gpu_allowed("rope")
        || s1.dtype != DType::F32
        || s2.dtype != DType::F32
        || s3.dtype != DType::F32
    {
        return Ok(None);
    }
    if !l1.is_contiguous() || !l2.is_contiguous() || !l3.is_contiguous() {
        return Ok(None);
    }
    let (b, h, t, d) = l1.shape().dims4()?;
    if d % 2 != 0 || h == 0 || t == 0 {
        return Ok(None);
    }
    // candle treats 3-dimensional cos/sin as per-batch rather than shared.
    let unbatched = l2.dims().len() == 3 && l3.dims().len() == 3;
    let cs_b = if unbatched { t * (d / 2) } else { 0 };
    let out = s1.device.alloc_elems(b * h * t * d, DType::F32)?;
    unsafe {
        k::candle_offload_rope_f32(
            s1.at(l1.start_offset()),
            s2.at(l2.start_offset()),
            s3.at(l3.start_offset()),
            out.ptr(),
            b * h,
            t,
            d,
            h,
            cs_b,
        )
    };
    ffi::note_launch();
    ffi::note_gpu_op();
    Ok(Some((out, l1.shape().clone())))
}

/// dtype conversion. Only the four float pairs that a model load actually
/// exercises are handled; everything else round-trips through the CPU.
pub(crate) fn to_dtype(
    s: &OffloadStorage,
    l: &Layout,
    dtype: DType,
) -> Result<Option<OffloadStorage>> {
    if !gpu_allowed("to_dtype") || !l.is_contiguous() {
        return Ok(None);
    }
    let op = match (s.dtype, dtype) {
        (DType::F16, DType::F32) => k::CVT_F16_F32,
        (DType::BF16, DType::F32) => k::CVT_BF16_F32,
        (DType::F32, DType::F16) => k::CVT_F32_F16,
        (DType::F32, DType::BF16) => k::CVT_F32_BF16,
        _ => return Ok(None),
    };
    let n = l.shape().elem_count();
    let out = s.device.alloc_elems(n, dtype)?;
    unsafe { k::candle_offload_convert(s.at(l.start_offset()), out.ptr(), n, op) };
    ffi::note_launch();
    ffi::note_gpu_op();
    Ok(Some(out))
}

/// Embedding lookup and friends. Without this the whole (vocab x hidden) table
/// is downloaded once per token, which dominated the transfer budget.
pub(crate) fn index_select(
    src: &OffloadStorage,
    ids: &OffloadStorage,
    l: &Layout,
    ids_l: &Layout,
    dim: usize,
) -> Result<Option<OffloadStorage>> {
    if !gpu_allowed("index_select") || src.dtype != DType::F32 {
        return Ok(None);
    }
    if !l.is_contiguous() || !ids_l.is_contiguous() {
        return Ok(None);
    }
    let ids_kind = match ids.dtype {
        DType::U32 => k::IDS_U32,
        DType::I64 => k::IDS_I64,
        _ => return Ok(None),
    };
    let dims = l.shape().dims();
    if dim >= dims.len() {
        return Ok(None);
    }
    let left: usize = dims[..dim].iter().product();
    let mid = dims[dim];
    let right: usize = dims[dim + 1..].iter().product();
    let n_ids = ids_l.shape().elem_count();
    if right == 0 || n_ids == 0 {
        return Ok(None);
    }
    let out = src.device.alloc_elems(left * n_ids * right, DType::F32)?;
    unsafe {
        k::candle_offload_index_select_f32(
            src.at(l.start_offset()),
            ids.at(ids_l.start_offset()),
            out.ptr(),
            left,
            n_ids,
            right,
            mid,
            ids_kind,
        )
    };
    ffi::note_launch();
    ffi::note_gpu_op();
    Ok(Some(out))
}
