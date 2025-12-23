use crate::{cudnn, WithDType};
use cudarc;
use cudarc::cudnn::safe::{ConvForward, Cudnn};
use cudarc::cudnn::ConvBiasActivationForward;
use cudarc::driver::{CudaSlice, CudaView, DeviceRepr, ValidAsZeroBits};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// The cudnn handles are stored per thread here rather than on the CudaDevice as they are neither
// send nor sync.
thread_local! {
    static CUDNN: RefCell<HashMap<crate::cuda_backend::DeviceId, Arc<Cudnn>>> = HashMap::new().into();
}

fn cudnn_handle(dev: &crate::cuda_backend::CudaDevice) -> crate::Result<Arc<Cudnn>> {
    let device_id = dev.id();
    Ok(CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(dev.cuda_stream());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        c
    })?)
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

    let cudnn = cudnn_handle(dev)?;
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

pub(crate) fn launch_conv2d_bias_activation<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    bias: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv2D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    use crate::conv::CudnnFwdAlgo as CandleAlgo;
    use cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t as A;

    let cudnn = cudnn_handle(dev)?;
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
    let act: cudarc::cudnn::ActivationDescriptor<T> = cudnn.create_activation(
        cudarc::cudnn::sys::cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY,
        cudarc::cudnn::sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        1.0,
    )?;
    // z has same shape as bias, but it's all 0s always, since we don't use the z-term.
    let z = cudnn.create_4d_tensor(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [1, params.c_out as i32, 1, 1],
        // [params.b_size as i32, params.c_out as i32, h_out, w_out],
    )?;
    let z_src = dev.alloc_zeros(params.c_out)?;
    let bias_d = cudnn.create_4d_tensor(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [1, params.c_out as i32, 1, 1],
    )?;
    let conv2d = ConvBiasActivationForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
        act: &act,
        z: &z,
        bias: &bias_d,
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
            &z_src.as_view(),
            bias,
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

    let cudnn = cudnn_handle(dev)?;
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
