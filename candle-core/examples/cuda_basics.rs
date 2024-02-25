#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{Device, Module, Tensor};

use candle_core::quantized::{QMatMul, QTensor};

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::randn(0f32, 1.0, (72, 32), &device)?;
    let q = QTensor::quantize(&q, candle_core::quantized::GgmlDType::Q4_0)?;
    let q = QMatMul::from_qtensor(q)?;
    let x = Tensor::randn(0f32, 1.0, (5, 32), &device)?;
    let res = q.forward(&x)?;
    println!("{res}");
    Ok(())
}
