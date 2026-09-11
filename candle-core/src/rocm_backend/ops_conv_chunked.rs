//! Bounded-memory im2col convolution.
//!
//! The single-shot path in [`super::ops_conv`] materializes the whole
//! `(b·h_out·w_out, c_in·k_h·k_w)` matrix before its one GEMM. Past a few
//! hundred MB that buffer is a memory-bandwidth cliff, and at Stable
//! Diffusion decode shapes it out-and-out OOMs (19.3 GB at 1024² for a 3×3
//! over 512 channels — reported on PR #3801). This module runs the same
//! im2col + GEMM in row chunks: unfold at most [`IM2COL_MAX_BYTES`] worth of
//! output rows at a time, GEMM each chunk, and write the rows into one
//! pre-allocated result. GEMM throughput is preserved — only the unfolded
//! buffer is bounded.
//!
//! The chunk kernels (`im2col_chunk_*`, `im2col1d_chunk_*`) are the plain
//! im2col kernels plus a base offset; they write chunk-local buffers.

use std::ffi::c_void;

use super::launch::launch_dense;
use super::params::info_buffer;
use super::{kernels, try_kernel_name, Map1, RocmDevice, RocmStorage, SendSyncDeviceMemory};
use crate::backend::{BackendDevice, BackendStorage};
use crate::{Layout, Result, Shape};

/// Ceiling on the materialized im2col buffer, in bytes.
///
/// Big enough that no LLM/transformer shape ever chunks (their convs are
/// small); small enough that conv-heavy models stay O(input) in memory. The
/// convolutions themselves get slightly faster past the cliff, so the exact
/// value is not sensitive.
pub(super) const IM2COL_MAX_BYTES: usize = 256 << 20;

/// im2col of `rows` output pixels `(b, h_out, w_out)` starting at `row0`,
/// into a `(rows, c_in·h_k·w_k)` chunk-local buffer.
struct Im2ColChunk<'a> {
    im: &'a super::ops_conv::Im2Col,
    row0: usize,
    rows: usize,
}

impl Map1 for Im2ColChunk<'_> {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let dims = layout.shape().dims();
        let (h_out, w_out) = self.im.hw_out(dims[2], dims[3]);
        let k = dims[1] * self.im.h_k * self.im.w_k;
        let chunk_numel = self.rows * k;
        let dst_base = self.row0 * k;
        let ds = info_buffer(dev, &[dims, layout.stride()])?;
        let dst = dev.alloc::<T>(chunk_numel)?;
        unsafe {
            let src_ptr = src.ptr_at(layout.start_offset());
            let dst_ptr = dst.as_ptr();
            let ds_ptr = ds.as_ptr();
            launch_dense(
                dev,
                &kernels::CONV,
                &try_kernel_name::<T>("im2col_chunk")?,
                chunk_numel,
                &mut [
                    &chunk_numel as *const usize as *mut c_void,
                    &dst_base as *const usize as *mut c_void,
                    &h_out as *const usize as *mut c_void,
                    &w_out as *const usize as *mut c_void,
                    &self.im.h_k as *const usize as *mut c_void,
                    &self.im.w_k as *const usize as *mut c_void,
                    &self.im.stride as *const usize as *mut c_void,
                    &self.im.padding as *const usize as *mut c_void,
                    &self.im.dilation as *const usize as *mut c_void,
                    (&ds_ptr) as *const *const usize as *mut c_void,
                    (&src_ptr) as *const *mut c_void as *mut c_void,
                    (&dst_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok(dst)
    }
}

/// im2col1d of `rows` output positions `(b, l_out)` starting at `row0`, into
/// a `(rows, c_in·l_k)` chunk-local buffer. One thread per `(b, l_out, c_in)`
/// triple, matching the un-chunked kernel.
struct Im2Col1DChunk<'a> {
    im: &'a super::ops_conv::Im2Col1D,
    row0: usize,
    rows: usize,
}

impl Map1 for Im2Col1DChunk<'_> {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let dims = layout.shape().dims();
        let l_out = self.im.l_out(dims[2]);
        let c_in = dims[1];
        let chunk_threads = self.rows * c_in;
        let thread_base = self.row0 * c_in;
        let ds = info_buffer(dev, &[dims, layout.stride()])?;
        let dst = dev.alloc::<T>(chunk_threads * self.im.l_k)?;
        unsafe {
            let src_ptr = src.ptr_at(layout.start_offset());
            let dst_ptr = dst.as_ptr();
            let ds_ptr = ds.as_ptr();
            launch_dense(
                dev,
                &kernels::CONV,
                &try_kernel_name::<T>("im2col1d_chunk")?,
                chunk_threads,
                &mut [
                    &chunk_threads as *const usize as *mut c_void,
                    &thread_base as *const usize as *mut c_void,
                    &l_out as *const usize as *mut c_void,
                    &self.im.l_k as *const usize as *mut c_void,
                    &self.im.stride as *const usize as *mut c_void,
                    &self.im.padding as *const usize as *mut c_void,
                    &self.im.dilation as *const usize as *mut c_void,
                    (&ds_ptr) as *const *const usize as *mut c_void,
                    (&src_ptr) as *const *mut c_void as *mut c_void,
                    (&dst_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok(dst)
    }
}

/// How many `(row, k)` rows fit under `max_col_bytes`, at least one.
fn rows_per_chunk(k: usize, dtype_bytes: usize, max_col_bytes: usize) -> usize {
    (max_col_bytes / (k * dtype_bytes)).max(1)
}

pub(super) fn conv2d(
    inp: &RocmStorage,
    l: &Layout,
    kernel: &RocmStorage,
    kernel_l: &Layout,
    params: &crate::conv::ParamsConv2D,
    max_col_bytes: usize,
) -> Result<RocmStorage> {
    let device = inp.device().clone();
    let im = super::ops_conv::Im2Col {
        h_k: params.k_h,
        w_k: params.k_w,
        stride: params.stride,
        dilation: params.dilation,
        padding: params.padding,
    };
    let h_out = params.out_h();
    let w_out = params.out_w();
    let b = params.b_size;
    let n = params.c_out;
    let k = params.k_h * params.k_w * params.c_in;
    let rows_total = b * h_out * w_out;
    let step = rows_per_chunk(k, inp.dtype().size_in_bytes(), max_col_bytes);

    let (rhs_owned, rhs_l) = super::ops_conv::kernel_operand(kernel, kernel_l, n, k)?;
    let rhs = rhs_owned.as_ref().unwrap_or(kernel);
    let mut res = unsafe { device.alloc_uninit(&Shape::from((rows_total, n)), inp.dtype())? };
    let mut row0 = 0;
    while row0 < rows_total {
        let rows = step.min(rows_total - row0);
        let col = RocmStorage {
            slice: Im2ColChunk {
                im: &im,
                row0,
                rows,
            }
            .map(&inp.slice, &device, l)?,
            device: device.clone(),
        };
        let col_l = Layout::contiguous((rows, k));
        let chunk = col.matmul(rhs, (1, rows, n, k), &col_l, &rhs_l)?;
        chunk.copy_strided_src(&mut res, row0 * n, &Layout::contiguous((rows, n)))?;
        row0 += rows;
    }

    let res_l = Layout::contiguous((b, h_out, w_out, n))
        .transpose(1, 2)?
        .transpose(1, 3)?;
    let mut res_t = unsafe { device.alloc_uninit(res_l.shape(), res.dtype())? };
    res.copy_strided_src(&mut res_t, 0, &res_l)?;
    Ok(res_t)
}

pub(super) fn conv1d(
    inp: &RocmStorage,
    l: &Layout,
    kernel: &RocmStorage,
    kernel_l: &Layout,
    params: &crate::conv::ParamsConv1D,
    max_col_bytes: usize,
) -> Result<RocmStorage> {
    let device = inp.device().clone();
    let im = super::ops_conv::Im2Col1D {
        l_k: params.k_size,
        stride: params.stride,
        dilation: params.dilation,
        padding: params.padding,
    };
    let l_out = params.l_out();
    let b = params.b_size;
    let n = params.c_out;
    let k = params.k_size * params.c_in;
    let rows_total = b * l_out;
    let step = rows_per_chunk(k, inp.dtype().size_in_bytes(), max_col_bytes);

    let (rhs_owned, rhs_l) = super::ops_conv::kernel_operand(kernel, kernel_l, n, k)?;
    let rhs = rhs_owned.as_ref().unwrap_or(kernel);
    let mut res = unsafe { device.alloc_uninit(&Shape::from((rows_total, n)), inp.dtype())? };
    let mut row0 = 0;
    while row0 < rows_total {
        let rows = step.min(rows_total - row0);
        let col = RocmStorage {
            slice: Im2Col1DChunk {
                im: &im,
                row0,
                rows,
            }
            .map(&inp.slice, &device, l)?,
            device: device.clone(),
        };
        let col_l = Layout::contiguous((rows, k));
        let chunk = col.matmul(rhs, (1, rows, n, k), &col_l, &rhs_l)?;
        chunk.copy_strided_src(&mut res, row0 * n, &Layout::contiguous((rows, n)))?;
        row0 += rows;
    }

    let res_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
    let mut res_t = unsafe { device.alloc_uninit(res_l.shape(), res.dtype())? };
    res.copy_strided_src(&mut res_t, 0, &res_l)?;
    Ok(res_t)
}
