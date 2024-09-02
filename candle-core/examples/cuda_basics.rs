#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
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
