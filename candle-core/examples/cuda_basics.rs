#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

const USE_CUDA_GRAPH: bool = true;

fn cuda_graph() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cu_device = match &device {
        Device::Cuda(dev) => dev,
        _ => unreachable!(),
    };
    let cu_stream = cu_device.cuda_stream();
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
        let _x = x.affine(1., 0.5)?;
        x.slice_set(&u, 0, 0)?;
        device.synchronize()?;
    }
    if USE_CUDA_GRAPH {
        cu_stream.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )?;
    }
    {
        let u = Tensor::zeros((4096, 4096), candle_core::DType::F32, &device)?
            .to_dtype(candle_core::DType::BF16)?;
        let mut x = Tensor::zeros((4096, 4096), candle_core::DType::F32, &device)?
            .to_dtype(candle_core::DType::BF16)?;
        let v = Tensor::zeros((4096, 1), candle_core::DType::F32, &device)?
            .to_dtype(candle_core::DType::BF16)?;
        for _i in 0..100 {
            // x.slice_set(&u, 0, 0)?;
            // x.broadcast_add(&v)?;
            x = x.affine(1., 0.5)?;
            // x = (&u + &x)?;
        }
    }
    if USE_CUDA_GRAPH {
        println!("capturing graph");
        let cu_graph = match cu_stream.end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY
        )? {
            None => anyhow::bail!("no graph captured"),
            Some(cu_graph) => cu_graph,
        };
        println!("graph captured!");
        for i in 1..100 {
            println!("graph exec {i}");
            cu_graph.launch()?;
            println!("sync");
            if let Err(err) = device.synchronize() {
                println!("err: {err:?}")
            }
            println!("done syncing");
        }
    } else {
        device.synchronize()?;
    }
    Ok(())
}

fn main() -> Result<()> {
    cuda_graph()?;
    return Ok(());
}

fn _matmul() -> Result<()> {
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
