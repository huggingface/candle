//! Direct (naive) convolution launchers from `candle-kernels/src/conv.cu`.
//!
//! These are the fallbacks the im2col path in [`super::ops_conv`] drops to when
//! a GEMM cannot express the operation — `conv_transpose1d` with padding or
//! dilation, and `conv_transpose2d` always. They read the input and the kernel
//! through the strides in `info`, so no operand has to be contiguous.
//!
//! `info` is `[src_dims, src_strides, k_dims, k_strides]`; the templates slice
//! it at fixed offsets (`info + 3`/`+ 6`/`+ 9` for 1-D, `info + 4`/`+ 8`/`+ 12`
//! for 2-D), so a 3-D input must be padded to rank 3 and a 4-D one to rank 4.

use std::ffi::c_void;

use super::utils::{info_buffer, launch_dense};
use super::{kernels, try_kernel_name, Map2, RocmDevice, SendSyncDeviceMemory};
use crate::{Layout, Result};

/// `conv1d`: `(b, c_in, l_in)` * `(c_out, c_in, k_size)`.
///
/// `CONV1D_OP`'s second parameter is named `num_dims` in the macro but is the
/// output length `l_out` in the template it forwards to.
pub(super) struct Conv1D<'a>(pub(super) &'a crate::conv::ParamsConv1D);

impl Map2 for Conv1D<'_> {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        inp: &SendSyncDeviceMemory<T>,
        inp_l: &Layout,
        k: &SendSyncDeviceMemory<T>,
        k_l: &Layout,
        dev: &RocmDevice,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let p = self.0;
        let dims = inp_l.shape().dims();
        if dims.len() != 3 {
            crate::bail!("unexpected input shape for conv1d {dims:?}")
        }
        let el = inp_l.shape().elem_count();
        let l_out = p.l_out();
        let dst_el = p.c_out * l_out * p.b_size;
        let ds = info_buffer(dev, &[dims, inp_l.stride(), k_l.dims(), k_l.stride()])?;
        let out = dev.alloc::<T>(dst_el)?;
        unsafe {
            let inp_ptr = inp.ptr_at(inp_l.start_offset());
            let k_ptr = k.ptr_at(k_l.start_offset());
            let out_ptr = out.as_ptr();
            let ds_ptr = ds.as_ptr();
            launch_dense(
                dev,
                &kernels::CONV,
                &try_kernel_name::<T>("conv1d")?,
                dst_el,
                &mut [
                    &el as *const usize as *mut c_void,
                    &l_out as *const usize as *mut c_void,
                    &p.stride as *const usize as *mut c_void,
                    &p.padding as *const usize as *mut c_void,
                    &p.dilation as *const usize as *mut c_void,
                    (&ds_ptr) as *const *const usize as *mut c_void,
                    (&inp_ptr) as *const *mut c_void as *mut c_void,
                    (&k_ptr) as *const *mut c_void as *mut c_void,
                    (&out_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok(out)
    }
}

/// `conv2d`: `(b, c_in, h_in, w_in)` * `(c_out, c_in, h_k, w_k)`.
///
/// `CONV2D_OP` takes `w_out` **before** `h_out`.
pub(super) struct Conv2D<'a>(pub(super) &'a crate::conv::ParamsConv2D);

impl Map2 for Conv2D<'_> {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        inp: &SendSyncDeviceMemory<T>,
        inp_l: &Layout,
        k: &SendSyncDeviceMemory<T>,
        k_l: &Layout,
        dev: &RocmDevice,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let p = self.0;
        let dims = inp_l.shape().dims();
        if dims.len() != 4 {
            crate::bail!("unexpected input shape for conv2d {dims:?}")
        }
        let (out_w, out_h) = (p.out_w(), p.out_h());
        let el = inp_l.shape().elem_count();
        let dst_el = p.c_out * out_w * out_h * p.b_size;
        let ds = info_buffer(dev, &[dims, inp_l.stride(), k_l.dims(), k_l.stride()])?;
        let out = dev.alloc::<T>(dst_el)?;
        unsafe {
            let inp_ptr = inp.ptr_at(inp_l.start_offset());
            let k_ptr = k.ptr_at(k_l.start_offset());
            let out_ptr = out.as_ptr();
            let ds_ptr = ds.as_ptr();
            launch_dense(
                dev,
                &kernels::CONV,
                &try_kernel_name::<T>("conv2d")?,
                dst_el,
                &mut [
                    &el as *const usize as *mut c_void,
                    &out_w as *const usize as *mut c_void,
                    &out_h as *const usize as *mut c_void,
                    &p.stride as *const usize as *mut c_void,
                    &p.padding as *const usize as *mut c_void,
                    &p.dilation as *const usize as *mut c_void,
                    (&ds_ptr) as *const *const usize as *mut c_void,
                    (&inp_ptr) as *const *mut c_void as *mut c_void,
                    (&k_ptr) as *const *mut c_void as *mut c_void,
                    (&out_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok(out)
    }
}

/// `conv_transpose1d`: `(b, c_in, l_in)` * `(c_in, c_out, l_k)`.
pub(super) struct ConvTranspose1D<'a>(pub(super) &'a crate::conv::ParamsConvTranspose1D);

impl Map2 for ConvTranspose1D<'_> {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        inp: &SendSyncDeviceMemory<T>,
        inp_l: &Layout,
        k: &SendSyncDeviceMemory<T>,
        k_l: &Layout,
        dev: &RocmDevice,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let p = self.0;
        let dims = inp_l.shape().dims();
        if dims.len() != 3 {
            crate::bail!("unexpected input shape for conv_transpose1d {dims:?}")
        }
        let l_out = p.l_out();
        let el = inp_l.shape().elem_count();
        let dst_el = p.c_out * l_out * p.b_size;
        let ds = info_buffer(dev, &[dims, inp_l.stride(), k_l.dims(), k_l.stride()])?;
        let out = dev.alloc::<T>(dst_el)?;
        unsafe {
            let inp_ptr = inp.ptr_at(inp_l.start_offset());
            let k_ptr = k.ptr_at(k_l.start_offset());
            let out_ptr = out.as_ptr();
            let ds_ptr = ds.as_ptr();
            launch_dense(
                dev,
                &kernels::CONV,
                &try_kernel_name::<T>("conv_transpose1d")?,
                dst_el,
                &mut [
                    &el as *const usize as *mut c_void,
                    &l_out as *const usize as *mut c_void,
                    &p.stride as *const usize as *mut c_void,
                    &p.padding as *const usize as *mut c_void,
                    &p.output_padding as *const usize as *mut c_void,
                    &p.dilation as *const usize as *mut c_void,
                    (&ds_ptr) as *const *const usize as *mut c_void,
                    (&inp_ptr) as *const *mut c_void as *mut c_void,
                    (&k_ptr) as *const *mut c_void as *mut c_void,
                    (&out_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok(out)
    }
}

/// `conv_transpose2d`: `(b, c_in, h_in, w_in)` * `(c_in, c_out, h_k, w_k)`.
///
/// `CONVT2D_OP` takes `w_out` before `h_out`, and `out_padding` between
/// `padding` and `dilation`.
pub(super) struct ConvTranspose2D<'a>(pub(super) &'a crate::conv::ParamsConvTranspose2D);

impl Map2 for ConvTranspose2D<'_> {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        inp: &SendSyncDeviceMemory<T>,
        inp_l: &Layout,
        k: &SendSyncDeviceMemory<T>,
        k_l: &Layout,
        dev: &RocmDevice,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let p = self.0;
        let dims = inp_l.shape().dims();
        if dims.len() != 4 {
            crate::bail!("unexpected input shape for conv_transpose2d {dims:?}")
        }
        let (out_w, out_h) = (p.out_w(), p.out_h());
        let el = inp_l.shape().elem_count();
        let dst_el = p.c_out * out_w * out_h * p.b_size;
        let ds = info_buffer(dev, &[dims, inp_l.stride(), k_l.dims(), k_l.stride()])?;
        let out = dev.alloc::<T>(dst_el)?;
        unsafe {
            let inp_ptr = inp.ptr_at(inp_l.start_offset());
            let k_ptr = k.ptr_at(k_l.start_offset());
            let out_ptr = out.as_ptr();
            let ds_ptr = ds.as_ptr();
            launch_dense(
                dev,
                &kernels::CONV,
                &try_kernel_name::<T>("conv_transpose2d")?,
                dst_el,
                &mut [
                    &el as *const usize as *mut c_void,
                    &out_w as *const usize as *mut c_void,
                    &out_h as *const usize as *mut c_void,
                    &p.stride as *const usize as *mut c_void,
                    &p.padding as *const usize as *mut c_void,
                    &p.output_padding as *const usize as *mut c_void,
                    &p.dilation as *const usize as *mut c_void,
                    (&ds_ptr) as *const *const usize as *mut c_void,
                    (&inp_ptr) as *const *mut c_void as *mut c_void,
                    (&k_ptr) as *const *mut c_void as *mut c_void,
                    (&out_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok(out)
    }
}
