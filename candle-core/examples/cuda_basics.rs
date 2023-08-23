#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let t = Tensor::randn(0f32, 1f32, (2, 4, 96, 96), &device)?;
    let w = Tensor::randn(0f32, 1f32, (320, 4, 3, 3), &device)?;
    let res = t.conv2d(&w, 1, 1, 1)?;
    println!("{res:?}");
    Ok(())
}
