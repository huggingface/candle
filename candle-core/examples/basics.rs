#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let inp = Tensor::randn(0f32, 1., (2, 320, 96, 96), &Device::Cpu)?;
    let w = Tensor::randn(0f32, 1., (320, 320, 3, 3), &Device::Cpu)?;
    let start = std::time::Instant::now();
    let res = inp.conv2d(&w, 0, 1, 1)?;
    println!("{:?}", start.elapsed());
    println!("{res:?}");
    Ok(())
}
