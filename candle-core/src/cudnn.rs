use crate::WithDType;
use cudarc;
use cudarc::cudnn::safe::{Conv2dForward, Cudnn};
use cudarc::driver::{CudaSlice, CudaView, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

pub(crate) fn launch_conv2d<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv2D,
) -> Result<(), cudarc::cudnn::result::CudnnError> {
    let cudnn = Arc::new(Cudnn::new(dst.device())?);
    let conv = cudnn.create_conv2d::<T>(
        [params.padding as i32, params.padding as i32],
        [params.stride as i32, params.stride as i32],
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
    unsafe {
        conv2d.launch::<CudaSlice<u8>, _, _, _>(alg, None, (T::one(), T::zero()), src, filter, dst)
    }
}
