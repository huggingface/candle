use crate::WithDType;
use cudarc;
use cudarc::cudnn::safe::{Conv2dForward, Cudnn};
use cudarc::driver::{CudaSlice, CudaView, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

impl From<cudarc::cudnn::CudnnError> for crate::Error {
    fn from(err: cudarc::cudnn::CudnnError) -> Self {
        crate::Error::wrap(err)
    }
}

impl From<cudarc::driver::DriverError> for crate::Error {
    fn from(err: cudarc::driver::DriverError) -> Self {
        crate::Error::wrap(err)
    }
}

pub(crate) fn launch_conv2d<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv2D,
) -> crate::Result<()> {
    let cudnn = Arc::new(Cudnn::new(dst.device())?);
    let conv = cudnn.create_conv2d::<T>(
        /* pad */ [params.padding as i32, params.padding as i32],
        /* stride */ [params.stride as i32, params.stride as i32],
        /* dilation */ [1, 1],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    let x = cudnn.create_4d_tensor(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.b_size as i32,
            params.c_in as i32,
            params.i_w as i32,
            params.i_h as i32,
        ],
    )?;
    let w = cudnn.create_4d_filter(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.c_out as i32,
            params.c_in as i32,
            params.k_w as i32,
            params.k_h as i32,
        ],
    )?;
    let (w_out, h_out) = (params.out_w() as i32, params.out_h() as i32);
    let y = cudnn.create_4d_tensor(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [params.b_size as i32, params.c_out as i32, w_out, h_out],
    )?;
    let conv2d = Conv2dForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let alg = conv2d.pick_algorithm()?;
    let workspace_size = conv2d.get_workspace_size(alg)?;
    let mut workspace = dst.device().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        conv2d.launch::<CudaSlice<u8>, _, _, _>(
            alg,
            Some(&mut workspace),
            (T::one(), T::zero()),
            src,
            filter,
            dst,
        )?;
    }
    Ok(())
}
