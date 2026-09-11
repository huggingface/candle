//! The `q8_1` activation requantization both integer matmul paths share.
//!
//! [`super::mmvq`] and [`super::mmq`] differ only in what they do *after* this:
//! the activations go to `q8_1` once, and the dot products then run in integer
//! SIMD (`__dp4a`, one `v_dot4_i32_iu8` per step on gfx11+) against the packed
//! weights, rather than dequantizing those weights into a float matrix.

use super::kernels::{arg, launch_err, MATRIX_ROW_PADDING};
use crate::quantized::GgmlDType;
use crate::rocm_backend::rocm_rs::hip::Dim3;
use crate::rocm_backend::{kernels, RocmDevice, SendSyncDeviceMemory};
use crate::Result;

/// `quantize_q8_1`'s thread block (`CUDA_QUANTIZE_BLOCK_SIZE`). A block covers
/// 256 columns, i.e. eight `QK8_1` blocks, one per wave32.
const QUANTIZE_BLOCK_SIZE: usize = 256;

/// `gridDim.y` is capped at 65535, so a tall activation matrix is quantized in
/// chunks of rows. Only the MMQ-shaped calls ever come close, but the loop is
/// cheap and matches `quantized/cuda.rs`.
const MAX_GRID_Y: usize = 65535;

pub(super) fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

/// Set `CANDLE_ROCM_FORCE_DMMV=1` to disable every `q8_1` path — MMVQ and MMQ
/// both — leaving [`super::dmmv`] and the dense dequantize.
///
/// DMMV does not requantize the activations, so it is the reference when an
/// accuracy question comes up, and it is how the paths are benchmarked against
/// each other. `quantized/cuda.rs` gives its own `FORCE_DMMV` the same reach.
pub(super) fn force_dmmv() -> bool {
    static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCE.get_or_init(|| match std::env::var("CANDLE_ROCM_FORCE_DMMV") {
        Ok(v) => v != "0" && !v.is_empty(),
        Err(_) => false,
    })
}

/// Bytes a `q8_1` buffer of `rows` rows of `k` columns needs, padding included.
pub(super) fn buffer_bytes(k: usize, rows: usize) -> usize {
    rows * pad(k, MATRIX_ROW_PADDING) * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size()
}

/// Quantize `ky` rows of `k` f32 columns into `dst` as `q8_1`.
///
/// ```text
/// quantize_q8_1(const float *x, void *vy, const int kx, const int kx_padded)
/// ```
///
/// Rows are padded out to `MATRIX_ROW_PADDING` columns and the tail is zero
/// filled by the kernel itself (`const float xi = ix < kx ? x[iy*kx + ix] : 0`),
/// so `dst` must be sized for the *padded* width — [`buffer_bytes`]. `x` is
/// indexed as `iy*kx`, i.e. the source rows are packed at their unpadded width.
pub(super) fn quantize_q8_1(
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
