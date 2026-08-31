//! The shared-memory 2D transpose, and the layout test that routes to it.
//!
//! [`super::ops_copy`]'s generic `ucopy_*` will materialise a transposed view,
//! but strides one of its two sides by a whole row: `t()?.contiguous()` on a
//! 8192x2048 f16 matrix measured 3.4 ms on gfx1101, about 20 GB/s of the card's
//! 624. `transpose2d_*` in `unary.cu` stages a 32x32 tile through LDS instead
//! and moves 292-662 GB/s.
//!
//! Two callers: the `copy_strided_src` fast path below, and the quantized dense
//! fallback in [`crate::quantized::rocm`], which needs its dequantized weights
//! in the orientation rocBLAS is fast at.

use super::launch::launch_kernel;
use super::{kernels, RocmDevice, RocmStorage, RocmStorageSlice, SendSyncDeviceMemory};
use crate::{Layout, Result};

/// Mirrors `TRANSPOSE_TILE` and `TRANSPOSE_BLOCK_ROWS` in `unary.cu`.
const TILE: usize = 32;
const BLOCK_ROWS: usize = 8;

/// `(batch, rows, cols)` when `l` views a contiguous buffer with its last two
/// dimensions swapped — the view is `(batch, rows, cols)`, memory holds
/// `(batch, cols, rows)` — or `None` for any other layout.
///
/// Extent-1 dimensions carry an arbitrary stride and are skipped; every other
/// leading dimension has to sit exactly one matrix from its neighbour, or the
/// batch is not one launch.
pub(super) fn as_transpose2d(l: &Layout) -> Option<(usize, usize, usize)> {
    let (dims, strides) = (l.dims(), l.stride());
    let n = dims.len();
    if n < 2 {
        return None;
    }
    let (rows, cols) = (dims[n - 2], dims[n - 1]);
    if rows == 0 || cols == 0 || strides[n - 2] != 1 || strides[n - 1] != rows {
        return None;
    }
    let mut expect = rows * cols;
    let mut batch = 1;
    for i in (0..n - 2).rev() {
        if dims[i] == 1 {
            continue;
        }
        if strides[i] != expect {
            return None;
        }
        batch *= dims[i];
        expect *= dims[i];
    }
    Some((batch, rows, cols))
}

/// `batch` matrices of `rows x cols` from `src` into `batch` of `cols x rows`
/// in `dst`, both contiguous, offsets in elements.
///
/// The caller owns the bounds: each side must hold `batch * rows * cols`
/// elements past its offset.
#[allow(clippy::too_many_arguments)]
pub(crate) fn transpose2d<T>(
    dev: &RocmDevice,
    suffix: &str,
    src: &SendSyncDeviceMemory<T>,
    src_offset: usize,
    dst: &SendSyncDeviceMemory<T>,
    dst_offset: usize,
    batch: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    if batch == 0 || rows == 0 || cols == 0 {
        return Ok(());
    }
    // No candle tensor reaches these, but a short grid would silently leave
    // part of the output untouched, so they error rather than clamp.
    let grid = rocm_rs::hip::Dim3::new_3d(
        u32::try_from(cols.div_ceil(TILE))
            .map_err(|_| super::rocm_error(format!("transpose2d: {cols} columns is too many")))?,
        u32::try_from(rows.div_ceil(TILE))
            .map_err(|_| super::rocm_error(format!("transpose2d: {rows} rows is too many")))?,
        u32::try_from(batch).map_err(|_| {
            super::rocm_error(format!("transpose2d: batch of {batch} is too large"))
        })?,
    );
    let block = rocm_rs::hip::Dim3::new_2d(TILE as u32, BLOCK_ROWS as u32);
    let src_ptr = unsafe { src.ptr_at(src_offset) };
    let dst_ptr = unsafe { dst.ptr_at(dst_offset) };
    // SAFETY: the grid covers every tile, the kernel bounds checks both of its
    // accesses against `rows`/`cols`, and the caller sized both pointers.
    unsafe {
        launch_kernel(
            dev,
            &kernels::UNARY,
            &format!("transpose2d_{suffix}"),
            grid,
            block,
            &mut [
                &rows as *const usize as *mut std::ffi::c_void,
                &cols as *const usize as *mut std::ffi::c_void,
                (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                (&dst_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
            ],
        )
    }
}

/// [`transpose2d`] for the `copy_strided_src` fast path, which holds storages
/// rather than typed slices.
///
/// `false`, having launched nothing, when either allocation is shorter than the
/// copy — callers over-request, and the generic path applies the clamp.
pub(super) fn try_copy_transposed(
    src: &RocmStorage,
    dst: &mut RocmStorage,
    dst_offset: usize,
    src_l: &Layout,
) -> Result<bool> {
    let Some((batch, rows, cols)) = as_transpose2d(src_l) else {
        return Ok(false);
    };
    let el_count = batch * rows * cols;
    if src.slice.count().saturating_sub(src_l.start_offset()) < el_count
        || dst.slice.count().saturating_sub(dst_offset) < el_count
    {
        return Ok(false);
    }
    let src_offset = src_l.start_offset();

    macro_rules! run {
        ($variant:ident, $suffix:expr) => {{
            let (s, d) = match (&src.slice, &dst.slice) {
                (RocmStorageSlice::$variant(s), RocmStorageSlice::$variant(d)) => (s, d),
                _ => crate::bail!("dtype mismatch in copy_strided_src"),
            };
            // The kernel takes the `(batch, cols, rows)` memory actually
            // holds, not the `(batch, rows, cols)` the view claims.
            transpose2d(
                &src.device,
                $suffix,
                s,
                src_offset,
                d,
                dst_offset,
                batch,
                cols,
                rows,
            )?;
        }};
    }

    match &src.slice {
        RocmStorageSlice::U8(_) => run!(U8, "u8"),
        RocmStorageSlice::U32(_) => run!(U32, "u32"),
        RocmStorageSlice::I16(_) => run!(I16, "i16"),
        RocmStorageSlice::I32(_) => run!(I32, "i32"),
        RocmStorageSlice::I64(_) => run!(I64, "i64"),
        RocmStorageSlice::BF16(_) => run!(BF16, "bf16"),
        RocmStorageSlice::F16(_) => run!(F16, "f16"),
        RocmStorageSlice::F32(_) => run!(F32, "f32"),
        RocmStorageSlice::F64(_) => run!(F64, "f64"),
        RocmStorageSlice::F8E4M3(_) => run!(F8E4M3, "f8_e4m3"),
    }
    Ok(true)
}
