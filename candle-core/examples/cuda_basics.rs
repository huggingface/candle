#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

fn cuda_graph() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cu_device = match &device {
        Device::Cuda(dev) => dev,
        _ => unreachable!(),
    };
    let cu_stream = cu_device.cu_stream();
    {
        // load_ptx cannot be called while capturing the stream so we need this to happen
        // beforehand.
        let u = Tensor::zeros((4096, 4096), candle_core::DType::F32, &device)?
            .to_dtype(candle_core::DType::BF16)?;
        let mut x = Tensor::zeros((4096, 4096), candle_core::DType::F32, &device)?
            .to_dtype(candle_core::DType::BF16)?;
        let v = Tensor::zeros(4096, candle_core::DType::F32, &device)?
            .to_dtype(candle_core::DType::BF16)?;
        let _x = x.mul(&u)?.broadcast_add(&v)?;
        device.synchronize()?;
    }
    unsafe {
        cudarc::driver::sys::lib()
            .cuStreamBeginCapture_v2(
                *cu_stream,
                cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()?
    };
    {
        let u = Tensor::zeros((4096, 4096), candle_core::DType::F32, &device)?
            .to_dtype(candle_core::DType::BF16)?;
        let mut x = Tensor::zeros((4096, 4096), candle_core::DType::F32, &device)?
            .to_dtype(candle_core::DType::BF16)?;
        let v = Tensor::zeros(4096, candle_core::DType::F32, &device)?
            .to_dtype(candle_core::DType::BF16)?;
        for _i in 0..1 {
            x = x.mul(&u)?.broadcast_add(&v)?;
        }
    }
    let cu_graph = unsafe {
        let mut cu_graph = std::mem::MaybeUninit::uninit();
        cudarc::driver::sys::lib()
            .cuStreamEndCapture(*cu_stream, cu_graph.as_mut_ptr())
            .result()?;
        cu_graph.assume_init()
    };
    let cu_graph_e = unsafe {
        let mut cu_graph_e = std::mem::MaybeUninit::uninit();
        cudarc::driver::sys::lib()
            .cuGraphInstantiateWithFlags(cu_graph_e.as_mut_ptr(), cu_graph, 0)
            .result()?;
        cu_graph_e.assume_init()
    };
    for _i in 0..100 {
        unsafe {
            cudarc::driver::sys::lib()
                .cuGraphLaunch(cu_graph_e, *cu_stream)
                .result()?
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    cuda_graph()?;
    return Ok(());
    let device = Device::new_cuda_with_stream(0)?;
    let x = Tensor::randn(0f32, 1.0, (8 * 4096, 8 * 4096), &device)?
        .to_dtype(candle_core::DType::BF16)?;
    candle_core::cuda::set_gemm_reduced_precision_f32(false);
    candle_core::cuda::set_gemm_reduced_precision_bf16(false);
    let _x1 = x.matmul(&x)?;
    drop(_x1);
    let start_time = std::time::Instant::now();
    let _x1 = x.matmul(&x)?;
    device.synchronize()?;
    println!("fp32: {:?}", start_time.elapsed());
    drop(_x1);
    candle_core::cuda::set_gemm_reduced_precision_f32(true);
    candle_core::cuda::set_gemm_reduced_precision_bf16(true);
    let _x1 = x.matmul(&x)?;
    drop(_x1);
    let start_time = std::time::Instant::now();
    let _x1 = x.matmul(&x)?;
    device.synchronize()?;
    println!("tf32: {:?}", start_time.elapsed());
    drop(_x1);
    Ok(())
}
