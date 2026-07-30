//! Pooling and upsampling launchers from `candle-kernels/src/conv.cu`.
//!
//! All four kernels take `info = [src_dims, src_strides]` for a rank-4 input and
//! read the source through those strides, so a non-contiguous input is handled
//! without a copy. Their scalar prefixes do *not* agree:
//!
//! * `AVG_POOL2D_OP` / `MAX_POOL2D_OP` lead with `src_numel` (unused by the
//!   templates, which derive the output extent themselves) then `w_k, h_k,
//!   w_stride, h_stride`.
//! * `UPSAMPLE_NEAREST2D_OP` has no `numel` at all: `w_out, h_out` then two
//!   `double` scales.
//! * `UPSAMPLE_BILINEAR2D_OP` interleaves three `bool`s with two `double`s.

use std::ffi::c_void;

use super::utils::{info_buffer, launch_dense};
use super::{kernels, try_kernel_name, Map1, RocmDevice, SendSyncDeviceMemory};
use crate::{Layout, Result};

/// `[dims, strides]` of a rank-4 input, rejecting anything else.
fn pool_info(
    dev: &RocmDevice,
    l: &Layout,
    op: &'static str,
) -> Result<(super::ParamBuffer, Vec<usize>)> {
    let dims = l.shape().dims();
    if dims.len() != 4 {
        crate::bail!("unexpected input shape for {op} {dims:?}")
    }
    let ds = info_buffer(dev, &[dims, l.stride()])?;
    Ok((ds, dims.to_vec()))
}

pub(super) enum PoolOp {
    Max,
    Avg,
}

pub(super) struct Pool2D {
    pub(super) w_k: usize,
    pub(super) h_k: usize,
    pub(super) w_stride: usize,
    pub(super) h_stride: usize,
    pub(super) op: PoolOp,
}

impl Map1 for Pool2D {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        inp: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        inp_l: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let (ds, dims) = pool_info(dev, inp_l, "pool")?;
        let el = inp_l.shape().elem_count();
        let out_w = (dims[2] - self.w_k) / self.w_stride + 1;
        let out_h = (dims[3] - self.h_k) / self.h_stride + 1;
        let dst_el = out_w * out_h * dims[0] * dims[1];
        let kname = match self.op {
            PoolOp::Max => "max_pool2d",
            PoolOp::Avg => "avg_pool2d",
        };
        let out = dev.alloc::<T>(dst_el)?;
        unsafe {
            let inp_ptr = inp.ptr_at(inp_l.start_offset());
            let out_ptr = out.as_ptr();
            let ds_ptr = ds.as_ptr();
            launch_dense(
                dev,
                &kernels::CONV,
                &try_kernel_name::<T>(kname)?,
                dst_el,
                &mut [
                    &el as *const usize as *mut c_void,
                    &self.w_k as *const usize as *mut c_void,
                    &self.h_k as *const usize as *mut c_void,
                    &self.w_stride as *const usize as *mut c_void,
                    &self.h_stride as *const usize as *mut c_void,
                    (&ds_ptr) as *const *const usize as *mut c_void,
                    (&inp_ptr) as *const *mut c_void as *mut c_void,
                    (&out_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok(out)
    }
}

pub(super) struct UpsampleNearest2D(pub(super) usize, pub(super) usize);

impl Map1 for UpsampleNearest2D {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        inp: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        inp_l: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let (ds, dims) = pool_info(dev, inp_l, "upsample")?;
        let (out_w, out_h) = (self.0, self.1);
        let dst_el = out_w * out_h * dims[0] * dims[1];
        let scale_w = dims[2] as f64 / out_w as f64;
        let scale_h = dims[3] as f64 / out_h as f64;
        let out = dev.alloc::<T>(dst_el)?;
        unsafe {
            let inp_ptr = inp.ptr_at(inp_l.start_offset());
            let out_ptr = out.as_ptr();
            let ds_ptr = ds.as_ptr();
            launch_dense(
                dev,
                &kernels::CONV,
                &try_kernel_name::<T>("upsample_nearest2d")?,
                dst_el,
                &mut [
                    &out_w as *const usize as *mut c_void,
                    &out_h as *const usize as *mut c_void,
                    &scale_w as *const f64 as *mut c_void,
                    &scale_h as *const f64 as *mut c_void,
                    (&ds_ptr) as *const *const usize as *mut c_void,
                    (&inp_ptr) as *const *mut c_void as *mut c_void,
                    (&out_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok(out)
    }
}

pub(super) struct UpsampleBilinear2D {
    pub(super) out_w: usize,
    pub(super) out_h: usize,
    pub(super) align_corners: bool,
    pub(super) scale_h_factor: Option<f64>,
    pub(super) scale_w_factor: Option<f64>,
}

impl Map1 for UpsampleBilinear2D {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        inp: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        inp_l: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        let (ds, dims) = pool_info(dev, inp_l, "upsample_bilinear2d")?;
        let (out_w, out_h) = (self.out_w, self.out_h);
        let dst_el = out_w * out_h * dims[0] * dims[1];
        let has_scale_h = self.scale_h_factor.is_some();
        let has_scale_w = self.scale_w_factor.is_some();
        let scale_h = self.scale_h_factor.unwrap_or(0.0);
        let scale_w = self.scale_w_factor.unwrap_or(0.0);
        let align_corners = self.align_corners;
        let out = dev.alloc::<T>(dst_el)?;
        unsafe {
            let inp_ptr = inp.ptr_at(inp_l.start_offset());
            let out_ptr = out.as_ptr();
            let ds_ptr = ds.as_ptr();
            launch_dense(
                dev,
                &kernels::CONV,
                &try_kernel_name::<T>("upsample_bilinear2d")?,
                dst_el,
                &mut [
                    &out_w as *const usize as *mut c_void,
                    &out_h as *const usize as *mut c_void,
                    &align_corners as *const bool as *mut c_void,
                    &has_scale_h as *const bool as *mut c_void,
                    &scale_h as *const f64 as *mut c_void,
                    &has_scale_w as *const bool as *mut c_void,
                    &scale_w as *const f64 as *mut c_void,
                    (&ds_ptr) as *const *const usize as *mut c_void,
                    (&inp_ptr) as *const *mut c_void as *mut c_void,
                    (&out_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok(out)
    }
}
