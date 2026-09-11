//! The dequantize-then-GEMM fallback, and the orientation it hands rocBLAS.
//!
//! The dequantize kernels write the weights as `n` rows of `k`. Describing that
//! buffer as `(k, n)` with swapped strides — what this path did
//! unconditionally — costs nothing on the host but reaches `gemm_config` as
//! `Operation::Transpose`, and rocBLAS on gfx1101 is *erratic* in that form:
//! over the sweep in [`super::bench_prefill`] the same f16 GEMM ranges 6-45
//! TFLOP/s with no monotonicity in the batch, against 25-57 for the plain form.
//!
//! Reorienting first costs one pass over `n * k` through the backend's
//! shared-memory transpose. That is often a net loss against the transposed
//! form — up to 1.8x — but it is *predictable*, and only a predictable path can
//! be dispatched to ahead of MMQ. Worst cell over three dtypes and four
//! Qwen3.5-2B shapes, `mmq_ms` over the dense path (above 1.0, dense wins):
//!
//! ```text
//! batch                128    256    384    512   1024   2048
//! transposed rhs      1.21   0.51   1.62   1.46   0.71   0.72
//! reoriented rhs      0.85   1.19   1.38   1.55   1.36   1.90
//! ```
//!
//! The transposed row dips under 1.0 at three of six batches, so no threshold
//! makes it safe. The reoriented row clears 1.0 from 256 up, peaking at 3.02x.

use super::QRocmStorage;
use crate::backend::BackendStorage;
use crate::quantized::GgmlDType;
use crate::rocm_backend::{transpose2d, RocmStorage, RocmStorageSlice};
use crate::{DType, Layout, Result, Shape};

/// Take this path ahead of [`super::mmq`], reorienting the weights, from this
/// batch up.
///
/// MMQ allocates nothing, which is why it is the default, but it runs at ~28%
/// of the card's `dp4a` ceiling (20-22 TOP/s against roughly 75) where a
/// reoriented f16 GEMM reaches 45. 256 is the first batch where the dense path
/// wins *everywhere* — worst cell 1.19, against 0.85 at 128 — the same rule
/// `mmq::min_work` is set by.
///
/// The threshold moves with the transpose kernel: it sat at 384 before the
/// diagonal reindexing took that from 48-390 GB/s to 292-662. Re-measure it,
/// not just the ratios, if that kernel changes.
const MIN_BATCH: usize = 256;

/// Above this many weight elements, leave it to MMQ whatever the batch.
///
/// This path holds `n * k` twice at once — the dequantize's output and the
/// transpose's — so a 4096x32000 `lm_head` would want 512 MB of f16 transient
/// where MMQ wants none. 32 Mi is 128 MB: past every projection in a model this
/// card runs, short of any vocabulary-sized matrix.
const MAX_WEIGHTS: usize = 32 << 20;

/// Whether [`forward`] is the *preferred* path rather than the last resort,
/// which decides both that [`QRocmStorage::fwd`] skips MMQ and that the weights
/// are worth reorienting.
///
/// f32 is excluded: its GEMM peak is half f16's and its dequantize moves twice
/// the bytes, so MMQ stayed ahead at every shape measured. bf16 has no
/// dequantize kernel of its own and would pay those same two passes via f32.
pub(super) fn preferred(act: DType, dtype: GgmlDType, n: usize, k: usize, b_size: usize) -> bool {
    act == DType::F16
        && b_size >= MIN_BATCH
        && n * k <= MAX_WEIGHTS
        && super::kernels::has_dequantize_kernel(dtype)
}

/// `(rows, cols)` contiguous in, `(cols, rows)` contiguous out.
fn transposed(w: &RocmStorage, rows: usize, cols: usize) -> Result<RocmStorage> {
    let dev = &w.device;
    macro_rules! run {
        ($variant:ident, $rust_ty:ty, $suffix:expr, $src:expr) => {{
            let dst = dev.alloc::<$rust_ty>(rows * cols)?;
            transpose2d(dev, $suffix, $src, 0, &dst, 0, 1, rows, cols)?;
            RocmStorageSlice::$variant(dst)
        }};
    }
    let slice = match &w.slice {
        RocmStorageSlice::F32(s) => run!(F32, f32, "f32", s),
        RocmStorageSlice::F16(s) => run!(F16, half::f16, "f16", s),
        RocmStorageSlice::BF16(s) => run!(BF16, half::bf16, "bf16", s),
        slice => crate::bail!("quantized dense matmul cannot reorient {:?}", slice.dtype()),
    };
    Ok(RocmStorage {
        slice,
        device: dev.clone(),
    })
}

/// Dequantize the whole weight matrix and hand it to rocBLAS, in the
/// activation's dtype.
pub(super) fn forward(
    q: &QRocmStorage,
    (b, m, n, k): (usize, usize, usize, usize),
    storage: &RocmStorage,
    layout: &Layout,
) -> Result<RocmStorage> {
    // An f16 activation gets the dedicated f16 dequantize kernel rather than a
    // full f32 buffer plus a cast: half the peak memory, and one fewer pass
    // over `n * k` elements. bf16 has no such kernel, and going via f16 would
    // clip anything past 65504, so it keeps the f32 route.
    let weights = match storage.dtype() {
        DType::F32 => q.dequantize(n * k)?,
        DType::F16 => q.dequantize_f16(n * k)?,
        dtype @ (DType::BF16 | DType::F64) => q
            .dequantize(n * k)?
            .to_dtype(&Layout::contiguous((n, k)), dtype)?,
        dtype => crate::bail!("quantized matmul expects a float input, got {dtype:?}"),
    };

    // Only the calls this path was *chosen* for pay for the good orientation.
    // As a fallback it keeps the stride swap: the transpose is a fixed cost the
    // small batches here do not amortise, and these are the dtypes the sweep
    // never covered.
    if preferred(storage.dtype(), q.dtype, n, k, b * m) {
        let weights = transposed(&weights, n, k)?;
        let rhs_l = Layout::contiguous(Shape::from((k, n))).broadcast_as((b, k, n))?;
        return storage.matmul(&weights, (b, m, n, k), layout, &rhs_l);
    }
    let rhs_l = Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
    storage.matmul(&weights, (b, m, n, k), layout, &rhs_l)
}
