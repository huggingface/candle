#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let t = Tensor::new(&[[1f32, 2., 3., 4.2]], &device)?;
    let sum = t.sum_keepdim(0)?;
    println!("{sum}");
    let sum = t.sum_keepdim(1)?;
    println!("{sum}");
    Ok(())
}
