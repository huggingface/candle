//! im2col / col2im launchers and the `conv*` entry points.
//!
//! Argument order comes from the macros in `candle-kernels/src/conv.cu` and from
//! nowhere else. Two traps live in that file:
//!
//! * `CONV1D_OP`'s second parameter is *named* `num_dims` but is forwarded to
//!   the `conv1d` template's `l_out` slot — it is the output length, not a rank.
//! * `CONV2D_OP`/`CONVT2D_OP`/`IM2COL_OP` take `w_out` **before** `h_out` while
//!   `AVG_POOL2D_OP`/`MAX_POOL2D_OP` take `w_k` before `h_k`; nothing in the file
//!   is consistently h-major.
//!
//! Every kernel here maps one thread to one output element and returns early
//! past the end, so they launch through [`launch_dense`], which sizes the grid
//! in full rather than reusing `launch_config` — that helper is free to launch
//! fewer threads than there are elements, which is only sound for grid-stride
//! loops.

use std::ffi::c_void;

use super::launch::launch_dense;
use super::params::info_buffer;
use super::{kernels, try_kernel_name, Map1, Map2, RocmDevice, RocmStorage, SendSyncDeviceMemory};
use crate::backend::{BackendDevice, BackendStorage};
use crate::{Layout, Result};

/// `im2col1d`: unfolds `(b, c_in, l_in)` into `(b, l_out, c_in, l_k)`.
pub(super) struct Im2Col1D {
    pub(super) l_k: usize,
    pub(super) stride: usize,
    pub(super) dilation: usize,
    pub(super) padding: usize,
}

impl Im2Col1D {
    fn l_out(&self, l: usize) -> usize {
        (l + 2 * self.padding - self.dilation * (self.l_k - 1) - 1) / self.stride + 1
    }
}

impl Map1 for Im2Col1D {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let dims = layout.shape().dims();
        let l_out = self.l_out(dims[2]);
        // One thread per (b, l_out, c_in) triple; each writes `l_k` elements.
        let threads = dims[0] * l_out * dims[1];
        let ds = info_buffer(dev, &[dims, layout.stride()])?;
        let dst = dev.alloc::<T>(threads * self.l_k)?;
        unsafe {
            let src_ptr = src.ptr_at(layout.start_offset());
            let dst_ptr = dst.as_ptr();
            let ds_ptr = ds.as_ptr();
            launch_dense(
                dev,
                &kernels::CONV,
                &try_kernel_name::<T>("im2col1d")?,
                threads,
                &mut [
                    &threads as *const usize as *mut c_void,
                    &l_out as *const usize as *mut c_void,
                    &self.l_k as *const usize as *mut c_void,
                    &self.stride as *const usize as *mut c_void,
                    &self.padding as *const usize as *mut c_void,
                    &self.dilation as *const usize as *mut c_void,
                    (&ds_ptr) as *const *const usize as *mut c_void,
                    (&src_ptr) as *const *mut c_void as *mut c_void,
                    (&dst_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok(dst)
    }
}

/// `im2col`: unfolds `(b, c_in, h_in, w_in)` into `(b, h_out, w_out, c_in, h_k, w_k)`.
pub(super) struct Im2Col {
    pub(super) h_k: usize,
    pub(super) w_k: usize,
    pub(super) stride: usize,
    pub(super) dilation: usize,
    pub(super) padding: usize,
}

impl Im2Col {
    fn hw_out(&self, h: usize, w: usize) -> (usize, usize) {
        let h_out = (h + 2 * self.padding - self.dilation * (self.h_k - 1) - 1) / self.stride + 1;
        let w_out = (w + 2 * self.padding - self.dilation * (self.w_k - 1) - 1) / self.stride + 1;
        (h_out, w_out)
    }
}

impl Map1 for Im2Col {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let dims = layout.shape().dims();
        let (h_out, w_out) = self.hw_out(dims[2], dims[3]);
        let dst_el = dims[0] * h_out * w_out * dims[1] * self.h_k * self.w_k;
        let ds = info_buffer(dev, &[dims, layout.stride()])?;
        let dst = dev.alloc::<T>(dst_el)?;
        unsafe {
            let src_ptr = src.ptr_at(layout.start_offset());
            let dst_ptr = dst.as_ptr();
            let ds_ptr = ds.as_ptr();
            launch_dense(
                dev,
                &kernels::CONV,
                &try_kernel_name::<T>("im2col")?,
                dst_el,
                &mut [
                    &dst_el as *const usize as *mut c_void,
                    &h_out as *const usize as *mut c_void,
                    &w_out as *const usize as *mut c_void,
                    &self.h_k as *const usize as *mut c_void,
                    &self.w_k as *const usize as *mut c_void,
                    &self.stride as *const usize as *mut c_void,
                    &self.padding as *const usize as *mut c_void,
                    &self.dilation as *const usize as *mut c_void,
                    (&ds_ptr) as *const *const usize as *mut c_void,
                    (&src_ptr) as *const *mut c_void as *mut c_void,
                    (&dst_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok(dst)
    }
}

/// `col2im1d`: folds `(b, l_in, c_out, k_size)` back into `(b, c_out, l_out)`.
///
/// `COL2IM1D_OP` takes no `info` pointer at all — it derives every stride from
/// the scalars, so the source has to be contiguous.
pub(super) struct Col2Im1D {
    pub(super) stride: usize,
}

impl Map1 for Col2Im1D {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        col: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        l: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let (b_size, l_in, c_out, k_size) = l.shape().dims4()?;
        let stride = self.stride;
        let l_out = (l_in - 1) * stride + k_size;
        let dst_el = b_size * c_out * l_out;
        let im = dev.alloc::<T>(dst_el)?;
        unsafe {
            let col_ptr = col.ptr_at(l.start_offset());
            let im_ptr = im.as_ptr();
            launch_dense(
                dev,
                &kernels::CONV,
                &try_kernel_name::<T>("col2im1d")?,
                dst_el,
                &mut [
                    &dst_el as *const usize as *mut c_void,
                    &l_out as *const usize as *mut c_void,
                    &l_in as *const usize as *mut c_void,
                    &c_out as *const usize as *mut c_void,
                    &k_size as *const usize as *mut c_void,
                    &stride as *const usize as *mut c_void,
                    (&col_ptr) as *const *mut c_void as *mut c_void,
                    (&im_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok(im)
    }
}

/// Right-hand side of the im2col GEMM: the `(c_out, k)` kernel, transposed.
///
/// `None` means "reuse the caller's buffer"; the layout is always the GEMM's.
///
/// DELIBERATE DIVERGENCE FROM cuda_backend: when the kernel layout is not
/// contiguous, cuda_backend copies it into `kernel_c` and then hands the GEMM
/// the *original* buffer anyway (`cuda_backend/mod.rs` conv1d/conv2d), so a
/// strided or offset kernel is read as if it were packed. Pinned here by
/// `conv2d_grad_noncontiguous_kernel`; worth fixing upstream.
fn kernel_operand(
    kernel: &RocmStorage,
    kernel_l: &Layout,
    n: usize,
    k: usize,
) -> Result<(Option<RocmStorage>, Layout)> {
    if kernel_l.is_contiguous() {
        let l = Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
        return Ok((None, l));
    }
    let mut kernel_c = unsafe {
        kernel
            .device()
            .alloc_uninit(kernel_l.shape(), kernel.dtype())?
    };
    kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
    Ok((Some(kernel_c), Layout::contiguous((n, k)).transpose(0, 1)?))
}

pub(super) fn conv1d(
    inp: &RocmStorage,
    l: &Layout,
    kernel: &RocmStorage,
    kernel_l: &Layout,
    params: &crate::conv::ParamsConv1D,
) -> Result<RocmStorage> {
    const USE_IM2COL_CONV1D: bool = true;

    let device = inp.device().clone();
    if !USE_IM2COL_CONV1D {
        let slice = super::ops_conv_direct::Conv1D(params).map(
            &inp.slice,
            l,
            &kernel.slice,
            kernel_l,
            &device,
        )?;
        return Ok(RocmStorage { slice, device });
    }

    let col = Im2Col1D {
        l_k: params.k_size,
        stride: params.stride,
        dilation: params.dilation,
        padding: params.padding,
    }
    .map(&inp.slice, &device, l)?;
    let col = RocmStorage {
        slice: col,
        device: device.clone(),
    };
    let l_out = params.l_out();
    let b = params.b_size;
    let n = params.c_out;
    let k = params.k_size * params.c_in;
    let m = l_out;
    let col_l = Layout::contiguous((b * m, k));
    let (rhs_owned, rhs_l) = kernel_operand(kernel, kernel_l, n, k)?;
    let rhs = rhs_owned.as_ref().unwrap_or(kernel);
    let res = col.matmul(rhs, (1, b * m, n, k), &col_l, &rhs_l)?;
    let res_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
    let mut res_t = unsafe { device.alloc_uninit(res_l.shape(), res.dtype())? };
    res.copy_strided_src(&mut res_t, 0, &res_l)?;
    Ok(res_t)
}

pub(super) fn conv2d(
    inp: &RocmStorage,
    l: &Layout,
    kernel: &RocmStorage,
    kernel_l: &Layout,
    params: &crate::conv::ParamsConv2D,
) -> Result<RocmStorage> {
    const USE_IM2COL_CONV2D: bool = true;

    let device = inp.device().clone();
    if !USE_IM2COL_CONV2D {
        let slice = super::ops_conv_direct::Conv2D(params).map(
            &inp.slice,
            l,
            &kernel.slice,
            kernel_l,
            &device,
        )?;
        return Ok(RocmStorage { slice, device });
    }

    let col = Im2Col {
        h_k: params.k_h,
        w_k: params.k_w,
        stride: params.stride,
        dilation: params.dilation,
        padding: params.padding,
    }
    .map(&inp.slice, &device, l)?;
    let col = RocmStorage {
        slice: col,
        device: device.clone(),
    };
    let h_out = params.out_h();
    let w_out = params.out_w();
    let b = params.b_size;
    let n = params.c_out;
    let k = params.k_h * params.k_w * params.c_in;
    let m = h_out * w_out;
    let col_l = Layout::contiguous((b * m, k));
    let (rhs_owned, rhs_l) = kernel_operand(kernel, kernel_l, n, k)?;
    let rhs = rhs_owned.as_ref().unwrap_or(kernel);
    let res = col.matmul(rhs, (1, b * m, n, k), &col_l, &rhs_l)?;
    let res_l = Layout::contiguous((b, h_out, w_out, n))
        .transpose(1, 2)?
        .transpose(1, 3)?;
    let mut res_t = unsafe { device.alloc_uninit(res_l.shape(), res.dtype())? };
    res.copy_strided_src(&mut res_t, 0, &res_l)?;
    Ok(res_t)
}

pub(super) fn conv_transpose1d(
    inp: &RocmStorage,
    l: &Layout,
    kernel: &RocmStorage,
    kernel_l: &Layout,
    params: &crate::conv::ParamsConvTranspose1D,
) -> Result<RocmStorage> {
    const USE_COL2IM_CONV1D_TR: bool = true;

    let device = inp.device().clone();
    let can_use_col2im = kernel_l.is_contiguous()
        && params.dilation == 1
        && params.padding == 0
        && params.output_padding == 0;
    let slice = if USE_COL2IM_CONV1D_TR && can_use_col2im {
        let (b_size, c_in, l_in) = l.shape().dims3()?;
        let (c_in2, c_out, k_size) = kernel_l.shape().dims3()?;
        if c_in != c_in2 {
            crate::bail!(
                "convtr1d: shape mismatch on c_in {:?} {:?}",
                l.shape(),
                kernel_l.shape()
            )
        }
        let col = {
            // Merges the last two kernel dimensions, and broadcasts it over the
            // batch with a zero leading stride.
            let kernel_l_mm = Layout::new(
                (b_size, c_in, k_size * c_out).into(),
                vec![0, k_size * c_out, 1],
                kernel_l.start_offset(),
            );
            inp.matmul(
                kernel,
                (b_size, l_in, c_out * k_size, c_in),
                &l.transpose(1, 2)?,
                &kernel_l_mm,
            )?
        };
        let col_l = Layout::contiguous((b_size, l_in, c_out, k_size));
        Col2Im1D {
            stride: params.stride,
        }
        .map(&col.slice, &device, &col_l)?
    } else {
        super::ops_conv_direct::ConvTranspose1D(params).map(
            &inp.slice,
            l,
            &kernel.slice,
            kernel_l,
            &device,
        )?
    };
    Ok(RocmStorage { slice, device })
}

pub(super) fn conv_transpose2d(
    inp: &RocmStorage,
    l: &Layout,
    kernel: &RocmStorage,
    kernel_l: &Layout,
    params: &crate::conv::ParamsConvTranspose2D,
) -> Result<RocmStorage> {
    let device = inp.device().clone();
    let slice = super::ops_conv_direct::ConvTranspose2D(params).map(
        &inp.slice,
        l,
        &kernel.slice,
        kernel_l,
        &device,
    )?;
    Ok(RocmStorage { slice, device })
}
