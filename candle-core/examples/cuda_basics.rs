#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let in_t = Tensor::rand(-1f32, 1f32, (1, 3, 12, 7), &device)?;
    let k_t = Tensor::rand(-1f32, 1f32, (6, 3, 1, 1), &device)?;
    let out_t = in_t.conv2d(&k_t, 0, 1, 1)?;
    println!("{out_t}");
    let in_t = in_t.to_device(&Device::Cpu)?;
    let k_t = k_t.to_device(&Device::Cpu)?;
    let out_t2 = in_t.conv2d(&k_t, 0, 1, 1)?;
    let diff = (out_t.to_device(&Device::Cpu)? - out_t2)?
        .sqr()?
        .sum_all()?;
    println!("{diff}");

    let t = Tensor::randn(0f32, 1f32, (2, 4, 96, 96), &device)?;
    let w = Tensor::randn(0f32, 1f32, (320, 4, 3, 3), &device)?;
    let res = t.conv2d(&w, 1, 1, 1)?;
    println!("{res:?}");
    Ok(())
}
