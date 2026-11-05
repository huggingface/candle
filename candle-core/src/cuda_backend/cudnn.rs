use crate::WithDType;
use cudarc;
use cudarc::cudnn::safe::{ConvBackwardData, ConvBackwardFilter, ConvForward, Cudnn};
use cudarc::driver::{CudaSlice, CudaView, DeviceRepr, ValidAsZeroBits};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// The cudnn handles are stored per thread here rather than on the CudaDevice as they are neither
// send nor sync.
thread_local! {
    static CUDNN: RefCell<HashMap<crate::cuda_backend::DeviceId, Arc<Cudnn>>> = HashMap::new().into();
    static BWD_DATA_PLANS: RefCell<HashMap<BackwardKey, (cudarc::cudnn::sys::cudnnConvolutionBwdDataAlgo_t, usize)>> = HashMap::new().into();
    static BWD_FILTER_PLANS: RefCell<HashMap<BackwardKey, (cudarc::cudnn::sys::cudnnConvolutionBwdFilterAlgo_t, usize)>> = HashMap::new().into();
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct BackwardKey {
    device_id: crate::cuda_backend::DeviceId,
    b_size: usize,
    c_in: usize,
    c_out: usize,
    i_h: usize,
    i_w: usize,
    k_h: usize,
    k_w: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
}

impl BackwardKey {
    fn new(params: &crate::conv::ParamsConv2D, device_id: crate::cuda_backend::DeviceId) -> Self {
        Self {
            device_id,
            b_size: params.b_size,
            c_in: params.c_in,
            c_out: params.c_out,
            i_h: params.i_h,
            i_w: params.i_w,
            k_h: params.k_h,
            k_w: params.k_w,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        }
    }
}

#[cfg(test)]
pub(crate) fn clear_handle_cache() {
    CUDNN.with(|cudnn| cudnn.borrow_mut().clear());
    BWD_DATA_PLANS.with(|plans| plans.borrow_mut().clear());
    BWD_FILTER_PLANS.with(|plans| plans.borrow_mut().clear());
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

fn cudnn_for_device(dev: &crate::cuda_backend::CudaDevice) -> crate::Result<Arc<Cudnn>> {
    let device_id = dev.id();
    CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(dev.cuda_stream());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        c.map_err(Into::into)
    })
}

pub(crate) fn launch_conv2d_backward_data_f32(
    filter: &CudaView<f32>,
    grad: &CudaView<f32>,
    dst: &mut CudaSlice<f32>,
    params: &crate::conv::ParamsConv2D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    let cudnn = cudnn_for_device(dev)?;
    let conv = cudnn.create_conv2d::<f32>(
        [params.padding as i32, params.padding as i32],
        [params.stride as i32, params.stride as i32],
        [params.dilation as i32, params.dilation as i32],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    let dx = cudnn.create_4d_tensor::<f32>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.b_size as i32,
            params.c_in as i32,
            params.i_h as i32,
            params.i_w as i32,
        ],
    )?;
    let w = cudnn.create_4d_filter::<f32>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.c_out as i32,
            params.c_in as i32,
            params.k_h as i32,
            params.k_w as i32,
        ],
    )?;
    let dy = cudnn.create_4d_tensor::<f32>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.b_size as i32,
            params.c_out as i32,
            params.out_h() as i32,
            params.out_w() as i32,
        ],
    )?;
    let backward = ConvBackwardData {
        conv: &conv,
        dx: &dx,
        w: &w,
        dy: &dy,
    };
    let key = BackwardKey::new(params, dev.id());
    let (algorithm, workspace_size) = BWD_DATA_PLANS.with(|plans| -> crate::Result<_> {
        let cached = plans.borrow().get(&key).copied();
        if let Some(plan) = cached {
            return Ok(plan);
        }
        let algorithm = backward.pick_algorithm()?;
        let workspace_size = backward.get_workspace_size(algorithm)?;
        plans.borrow_mut().insert(key, (algorithm, workspace_size));
        Ok((algorithm, workspace_size))
    })?;
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        backward.launch(
            algorithm,
            Some(&mut workspace),
            (1.0f32, 0.0f32),
            dst,
            filter,
            grad,
        )?;
    }
    Ok(())
}

pub(crate) fn launch_conv2d_backward_filter_f32(
    src: &CudaView<f32>,
    grad: &CudaView<f32>,
    dst: &mut CudaSlice<f32>,
    params: &crate::conv::ParamsConv2D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    let cudnn = cudnn_for_device(dev)?;
    let conv = cudnn.create_conv2d::<f32>(
        [params.padding as i32, params.padding as i32],
        [params.stride as i32, params.stride as i32],
        [params.dilation as i32, params.dilation as i32],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    let x = cudnn.create_4d_tensor::<f32>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.b_size as i32,
            params.c_in as i32,
            params.i_h as i32,
            params.i_w as i32,
        ],
    )?;
    let dw = cudnn.create_4d_filter::<f32>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.c_out as i32,
            params.c_in as i32,
            params.k_h as i32,
            params.k_w as i32,
        ],
    )?;
    let dy = cudnn.create_4d_tensor::<f32>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.b_size as i32,
            params.c_out as i32,
            params.out_h() as i32,
            params.out_w() as i32,
        ],
    )?;
    let backward = ConvBackwardFilter {
        conv: &conv,
        x: &x,
        dw: &dw,
        dy: &dy,
    };
    let key = BackwardKey::new(params, dev.id());
    let (algorithm, workspace_size) = BWD_FILTER_PLANS.with(|plans| -> crate::Result<_> {
        let cached = plans.borrow().get(&key).copied();
        if let Some(plan) = cached {
            return Ok(plan);
        }
        let algorithm = backward.pick_algorithm()?;
        let workspace_size = backward.get_workspace_size(algorithm)?;
        plans.borrow_mut().insert(key, (algorithm, workspace_size));
        Ok((algorithm, workspace_size))
    })?;
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        backward.launch(
            algorithm,
            Some(&mut workspace),
            (1.0f32, 0.0f32),
            src,
            dst,
            grad,
        )?;
    }
    Ok(())
}

pub(crate) fn launch_conv2d<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv2D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    use crate::conv::CudnnFwdAlgo as CandleAlgo;
    use cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t as A;

    let cudnn = cudnn_for_device(dev)?;
    let conv = cudnn.create_conv2d::<Y>(
        /* pad */ [params.padding as i32, params.padding as i32],
        /* stride */ [params.stride as i32, params.stride as i32],
        /* dilation */ [params.dilation as i32, params.dilation as i32],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.i_h as i32,
        params.i_w as i32,
    ];
    // Note that `src` already starts at the proper offset.
    let x = if src_l.is_contiguous() {
        cudnn.create_4d_tensor::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            x_shape,
        )?
    } else {
        let s = src_l.stride();
        cudnn.create_4d_tensor_ex::<T>(
            x_shape,
            [s[0] as i32, s[1] as i32, s[2] as i32, s[3] as i32],
        )?
    };
    let w = cudnn.create_4d_filter::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.c_out as i32,
            params.c_in as i32,
            params.k_h as i32,
            params.k_w as i32,
        ],
    )?;
    let (w_out, h_out) = (params.out_w() as i32, params.out_h() as i32);
    let y = cudnn.create_4d_tensor::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [params.b_size as i32, params.c_out as i32, h_out, w_out],
    )?;
    let conv2d = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let alg = match params.cudnn_fwd_algo {
        None => conv2d.pick_algorithm()?,
        Some(CandleAlgo::ImplicitGemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        Some(CandleAlgo::ImplicitPrecompGemm) => {
            A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        }
        Some(CandleAlgo::Gemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        Some(CandleAlgo::Direct) => A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        Some(CandleAlgo::Fft) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        Some(CandleAlgo::FftTiling) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        Some(CandleAlgo::Winograd) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        Some(CandleAlgo::WinogradNonFused) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        Some(CandleAlgo::Count) => A::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
    };
    let workspace_size = conv2d.get_workspace_size(alg)?;
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
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

pub(crate) fn launch_conv1d<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv1D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    use crate::conv::CudnnFwdAlgo as CandleAlgo;
    use cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t as A;

    let cudnn = cudnn_for_device(dev)?;
    let conv = cudnn.create_conv2d::<Y>(
        /* pad */ [params.padding as i32, 0],
        /* stride */ [params.stride as i32, 1],
        /* dilation */ [params.dilation as i32, 1],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnnsettensornddescriptor
    // > Tensors are restricted to having at least 4 dimensions, and at most CUDNN_DIM_MAX
    // > dimensions (defined in cudnn.h). When working with lower dimensional data, it is
    // > recommended that the user create a 4D tensor, and set the size along unused dimensions
    // > to 1.
    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.l_in as i32,
        1,
    ];
    // Note that `src` already starts at the proper offset.
    let x = if src_l.is_contiguous() {
        cudnn.create_4d_tensor::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            x_shape,
        )?
    } else {
        let s = src_l.stride();
        cudnn.create_4d_tensor_ex::<T>(x_shape, [s[0] as i32, s[1] as i32, s[2] as i32, 1i32])?
    };
    let w = cudnn.create_4d_filter::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.c_out as i32,
            params.c_in as i32,
            params.k_size as i32,
            1,
        ],
    )?;
    let l_out = params.l_out() as i32;
    let y = cudnn.create_4d_tensor::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [params.b_size as i32, params.c_out as i32, l_out, 1],
    )?;
    let conv1d = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let alg = match params.cudnn_fwd_algo {
        None => conv1d.pick_algorithm()?,
        Some(CandleAlgo::ImplicitGemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        Some(CandleAlgo::ImplicitPrecompGemm) => {
            A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        }
        Some(CandleAlgo::Gemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        Some(CandleAlgo::Direct) => A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        Some(CandleAlgo::Fft) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        Some(CandleAlgo::FftTiling) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        Some(CandleAlgo::Winograd) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        Some(CandleAlgo::WinogradNonFused) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        Some(CandleAlgo::Count) => A::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
    };
    let workspace_size = conv1d.get_workspace_size(alg)?;
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        conv1d.launch::<CudaSlice<u8>, _, _, _>(
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
