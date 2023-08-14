use crate::WithDType;
use cudarc;
use cudarc::cudnn::safe::{Conv2dForward, Cudnn};
use cudarc::driver::{CudaSlice, CudaView, DeviceRepr, ValidAsZeroBits};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// The cudnn handles are stored per thread here rather than on the CudaDevice as they are neither
// send nor sync.
thread_local! {
    static CUDNN: RefCell<HashMap<crate::cuda_backend::DeviceId, Arc<Cudnn>>> = HashMap::new().into();
}

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
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv2D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    let device_id = dev.id();
    let cudnn = CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(dev.cuda_device());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        c
    })?;
    let conv = cudnn.create_conv2d::<T>(
        /* pad */ [params.padding as i32, params.padding as i32],
        /* stride */ [params.stride as i32, params.stride as i32],
        /* dilation */ [1, 1],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.i_w as i32,
        params.i_h as i32,
    ];
    // Note that `src` already starts at the proper offset.
    let x = if src_l.is_contiguous() {
        cudnn.create_4d_tensor(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            x_shape,
        )?
    } else {
        let s = src_l.stride();
        cudnn.create_4d_tensor_ex(
            x_shape,
            [s[0] as i32, s[1] as i32, s[2] as i32, s[3] as i32],
        )?
    };
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
    let mut workspace = dev.cuda_device().alloc_zeros::<u8>(workspace_size)?;
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
