//! MMVQ: quantize the activations to `q8_1`, then dot them against the packed
//! weights with integer SIMD (`__dp4a`, one `v_dot4_i32_iu8` per step on
//! gfx11+).
//!
//! This is the decode fast-path. Against [`super::dmmv`] it covers batches of
//! up to eight rather than one, and against the dense fallback it never
//! materialises the weight matrix in f32 — an `lm_head` of 32000x4096 is
//! 500 MB of transient at f32, moved twice, per matmul.
//!
//! Every kernel signature below is copied out of `candle-kernels/src/quantized.cu`
//! rather than inferred; the families in that file disagree about arity and
//! argument order, and a wrong guess is a silent wrong answer.

use super::kernels::{arg, launch_err, MATRIX_ROW_PADDING, WARP_SIZE};
use super::QRocmStorage;
use crate::backend::BackendStorage;
use crate::quantized::GgmlDType;
use crate::rocm_backend::rocm_rs::hip::Dim3;
use crate::rocm_backend::{
    kernels, RocmDevice, RocmStorage, RocmStorageSlice, SendSyncDeviceMemory,
};
use crate::{DType, Layout, Result};

/// `quantize_q8_1`'s thread block (`CUDA_QUANTIZE_BLOCK_SIZE`). A block covers
/// 256 columns, i.e. eight `QK8_1` blocks, one per wave32.
const QUANTIZE_BLOCK_SIZE: usize = 256;

/// `gridDim.y` is capped at 65535, so a tall activation matrix is quantized in
/// chunks of rows. Only `mul_mat_via_q8_1`-shaped calls ever come close, but the
/// loop is cheap and matches `quantized/cuda.rs`.
const MAX_GRID_Y: usize = 65535;

/// Largest batch an `mul_mat_vec_*_q8_1_cuda<N>` instantiation exists for.
pub(super) const MAX_BATCH: usize = 8;

fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

/// Set `CANDLE_ROCM_FORCE_DMMV=1` to route every decode-shaped matmul through
/// [`super::dmmv`] instead. DMMV does not requantize the activations, so it is
/// the reference when an accuracy question comes up, and it is how the two
/// paths are benchmarked against each other.
pub(super) fn force_dmmv() -> bool {
    static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCE.get_or_init(|| match std::env::var("CANDLE_ROCM_FORCE_DMMV") {
        Ok(v) => v != "0" && !v.is_empty(),
        Err(_) => false,
    })
}

/// Kernel-name stem for `dtype`, without the batch-size suffix.
///
/// Note the inconsistent casing: the K-quants spell the dtype `q4_K` while the
/// legacy quants spell it `q4_0`. Taken verbatim from `quantized.cu:3009`
/// onwards.
fn kernel_stem(dtype: GgmlDType) -> Option<&'static str> {
    let stem = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_cuda",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_cuda",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_cuda",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_cuda",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_cuda",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_cuda",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_cuda",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_cuda",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_cuda",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_cuda",
        _ => return None,
    };
    Some(stem)
}

/// Grid width and `blockDim.y` for a batch of `b_size`.
///
/// The kernel picks `nwarps` and `rows_per_cuda_block` from `ncols_y` at compile
/// time (`quantized.cu:2939`), and the launch has to agree with the choice it
/// made or the block reduction reads uninitialised shared memory. Only the
/// non-HIP arm of that `#if` is live here: it keys on `GGML_USE_HIPBLAS` plus a
/// bare `RDNA2`/`RDNA3`, none of which this build defines. Same table as
/// `quantized/cuda.rs::mul_mat_vec_via_q8_1`.
fn launch_shape(nrows: usize, b_size: usize) -> Option<(usize, usize)> {
    match b_size {
        1 => Some((nrows, 4)),
        2..=4 => Some((nrows.div_ceil(2), 4)),
        5..=MAX_BATCH => Some((nrows.div_ceil(2), 2)),
        _ => None,
    }
}

/// Whether `(dtype, nrows, b_size)` can go through MMVQ at all.
///
/// The parity condition is a bound check this vendored copy of the kernel is
/// missing. Upstream llama.cpp writes the result as
///
/// ```text
/// if (threadIdx.x < rows_per_cuda_block &&
///     (rows_per_cuda_block == 1 || row0 + int(threadIdx.x) < nrows_dst)) {
/// ```
///
/// while `quantized.cu:3002` has only the first half:
///
/// ```text
/// if (threadIdx.x < rows_per_cuda_block) {
///     dst[j*nrows_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x];
/// }
/// ```
///
/// For `b_size == 1`, `rows_per_cuda_block` is 1 and the grid is exactly `nrows`
/// blocks, so nothing overshoots. For `b_size >= 2` it is 2 and the grid is
/// `nrows.div_ceil(2)`; an odd `nrows` then makes the last block write
/// `dst[j*nrows + nrows]`, which for every `j` but the last is the *first output
/// of the next batch row* — silent corruption, not an out-of-bounds fault. Odd
/// `nrows` with a batch therefore falls back rather than being fixed here, since
/// the fix belongs in a `.cu` file shared with the CUDA backend.
fn supports(dtype: GgmlDType, nrows: usize, ncols: usize, b_size: usize) -> bool {
    kernel_stem(dtype).is_some()
        && launch_shape(nrows, b_size).is_some()
        && ncols.is_multiple_of(dtype.block_size())
        && ncols > 0
        && nrows > 0
        && (b_size == 1 || nrows.is_multiple_of(2))
}

/// Weight elements below which MMVQ's fixed cost outweighs what it saves, for
/// the dtypes where it is ever worth paying at all. See [`dmmv_is_faster`].
const MMVQ_MIN_WORK: usize = 8 << 20;

/// Whether a batch of one is better served by [`super::dmmv`].
///
/// MMVQ pays for a second kernel launch and a second allocation to build the
/// `q8_1` activation, roughly 14 µs of fixed cost on gfx1101, and only earns it
/// back on a large enough weight matrix. Measured at `m = 1`, f32 activations,
/// milliseconds per matmul, DMMV against MMVQ:
///
/// ```text
///  k     n       q8_0           q4_0           q4_K           q6_K
/// 2048  2048  0.0158 0.0179  0.0121 0.0161  0.0070 0.0181  0.0091 0.0196
/// 2048  4096  0.0227 0.0226  0.0226 0.0207  0.0108 0.0238  0.0153 0.0261
/// 4096  4096  0.0429 0.0261  0.0433 0.0222  0.0174 0.0329  0.0280 0.0370
/// 4096 32000  0.3716 0.2495  0.3487 0.1423  0.1572 0.1637  0.2235 0.2026
/// ```
///
/// Two things fall out. The legacy quants have a much weaker DMMV kernel than
/// the K-quants do, so MMVQ overtakes them at roughly 8 Mi weight elements and
/// is then up to 2.4x faster. The K-quant DMMV kernels are already good enough
/// that MMVQ never clearly wins — 10% at the 131 Mi `lm_head`, less or nothing
/// everywhere else — so they keep DMMV at every size.
///
/// None of this applies above a batch of one, or to an f16/bf16 activation:
/// there DMMV is not an option at all and the alternative is the dense
/// dequantize, which MMVQ beats by an order of magnitude.
fn dmmv_is_faster(dtype: GgmlDType, work: usize) -> bool {
    match dtype {
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => true,
        _ => work < MMVQ_MIN_WORK,
    }
}

/// Quantize `ky` rows of `k` f32 columns into `dst` as `q8_1`.
///
/// ```text
/// quantize_q8_1(const float *x, void *vy, const int kx, const int kx_padded)
/// ```
///
/// Rows are padded out to `MATRIX_ROW_PADDING` columns and the tail is zero
/// filled by the kernel itself (`const float xi = ix < kx ? x[iy*kx + ix] : 0`),
/// so `dst` must be sized for the *padded* width. `x` is indexed as `iy*kx`,
/// i.e. the source rows are packed at their unpadded width.
fn quantize_q8_1(
    src: &SendSyncDeviceMemory<f32>,
    src_offset: usize,
    dst: &SendSyncDeviceMemory<u8>,
    k: usize,
    ky: usize,
    dev: &RocmDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = kx_padded.div_ceil(QUANTIZE_BLOCK_SIZE);
    let dst_row_bytes = kx_padded / GgmlDType::Q8_1.block_size() * GgmlDType::Q8_1.type_size();
    let func = dev.get_or_load_func("quantize_q8_1", &kernels::QUANTIZED)?;
    let kx_i = k as i32;
    let kx_padded_i = kx_padded as i32;

    let mut rows_done = 0;
    while rows_done < ky {
        let rows = (ky - rows_done).min(MAX_GRID_Y);
        // `ptr_at` scales by the element size, so the f32 source advances by
        // elements and the u8 destination by bytes.
        let src_ptr = unsafe { src.ptr_at(src_offset + rows_done * k) };
        let dst_ptr = unsafe { dst.ptr_at(rows_done * dst_row_bytes) };
        let mut args = vec![arg(&src_ptr), arg(&dst_ptr), arg(&kx_i), arg(&kx_padded_i)];
        func.launch(
            Dim3::new_2d(num_blocks as u32, rows as u32),
            Dim3::new_1d(QUANTIZE_BLOCK_SIZE as u32),
            0,
            Some(dev.stream()),
            &mut args,
        )
        .map_err(|e| launch_err("quantize_q8_1", e))?;
        rows_done += rows;
    }
    Ok(())
}

/// `data` (the `(nrows, ncols)` packed weights) times `b_size` activation rows.
///
/// ```text
/// mul_mat_vec_*_q8_1_cuda<N>(const void *vx, const void *vy, float *dst,
///                            const int ncols_x, const int nrows_x,
///                            const int nrows_y, const int nrows_dst)
/// ```
///
/// `nrows_y` is the *padded* activation width, not `ncols_x`: the kernel derives
/// `blocks_per_col_y = nrows_y / QK8_1` and uses it as the row stride into the
/// q8_1 buffer, which `quantize_q8_1` laid out at the padded width.
#[allow(clippy::too_many_arguments)]
pub(super) fn mul_mat_vec_via_q8_1(
    data: &SendSyncDeviceMemory<u8>,
    data_len: usize,
    y: &SendSyncDeviceMemory<f32>,
    y_offset: usize,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &RocmDevice,
) -> Result<SendSyncDeviceMemory<f32>> {
    let data_elems = data_len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("quantized mmvq: data holds {data_elems} elems, need {ncols}x{nrows}")
    }
    let (stem, (nblocks, nwarps)) = match (kernel_stem(dtype), launch_shape(nrows, b_size)) {
        (Some(stem), Some(shape)) => (stem, shape),
        _ => crate::bail!("no ROCm mmvq kernel for {dtype:?} at batch {b_size}"),
    };

    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    // Zeroed rather than uninitialised: `quantize_q8_1` writes every byte of the
    // padded row, but a zeroed buffer keeps a future geometry change from
    // feeding the dot product garbage instead of an obvious zero.
    let y_q8_1 = dev.alloc_zeros::<u8>(y_bytes)?;
    quantize_q8_1(y, y_offset, &y_q8_1, ncols, b_size, dev)?;

    let name = format!("{stem}{b_size}");
    let func = dev.get_or_load_func(&name, &kernels::QUANTIZED)?;
    let dst = dev.alloc_zeros::<f32>(nrows * b_size)?;
    let src_ptr = data.as_ptr();
    let y_ptr = y_q8_1.as_ptr();
    let dst_ptr = dst.as_ptr();
    let ncols_x = ncols as i32;
    let nrows_x = nrows as i32;
    let nrows_y = ncols_padded as i32;
    let nrows_dst = nrows as i32;
    let mut args = vec![
        arg(&src_ptr),
        arg(&y_ptr),
        arg(&dst_ptr),
        arg(&ncols_x),
        arg(&nrows_x),
        arg(&nrows_y),
        arg(&nrows_dst),
    ];
    func.launch(
        Dim3::new_1d(nblocks as u32),
        Dim3::new_2d(WARP_SIZE as u32, nwarps as u32),
        0,
        Some(dev.stream()),
        &mut args,
    )
    .map_err(|e| launch_err(&name, e))?;
    Ok(dst)
}

/// The MMVQ dispatch for [`QRocmStorage::fwd`], or `None` when the shape,
/// dtype or activation is one this path does not cover and the caller should
/// fall through.
///
/// `n` is the number of output rows, `k` the shared dimension and `b_size` the
/// flattened batch (`b * m`). Result rows are `n` f32 values per batch row,
/// which is exactly the contiguous `(b_size, n)` output the caller wants.
pub(super) fn try_fwd(
    q: &QRocmStorage,
    n: usize,
    k: usize,
    b_size: usize,
    storage: &RocmStorage,
    layout: &Layout,
) -> Result<Option<SendSyncDeviceMemory<f32>>> {
    if force_dmmv() || !supports(q.dtype, n, k, b_size) {
        return Ok(None);
    }
    let (o1, o2) = match layout.contiguous_offsets() {
        Some(offsets) => offsets,
        None => return Ok(None),
    };
    if o2 - o1 != b_size * k {
        return Ok(None);
    }

    // `quantize_q8_1` reads f32, so an f16/bf16 activation is cast first. That
    // is `b_size * k` elements — three orders of magnitude below the `n * k`
    // weight matrix the dense fallback would otherwise have to materialise.
    let cast;
    let (y, y_offset) = match &storage.slice {
        RocmStorageSlice::F32(y) => {
            // The only case where a *cheaper* alternative exists: DMMV takes
            // exactly a batch of one, contiguous and f32.
            if b_size == 1 && super::dmmv::has_kernel(q.dtype) && dmmv_is_faster(q.dtype, n * k) {
                return Ok(None);
            }
            (y, o1)
        }
        RocmStorageSlice::F16(_) | RocmStorageSlice::BF16(_) => {
            cast = storage.to_dtype(layout, DType::F32)?;
            match &cast.slice {
                RocmStorageSlice::F32(y) => (y, 0usize),
                _ => return Ok(None),
            }
        }
        _ => return Ok(None),
    };

    let dst = mul_mat_vec_via_q8_1(
        &q.data, q.len, y, y_offset, q.dtype, k, n, b_size, &q.device,
    )?;
    Ok(Some(dst))
}

#[cfg(test)]
mod tests;
