#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

pub const N_ITERS: usize = 5;

fn main() -> Result<()> {
    let inp = Tensor::randn(0f32, 1., (1, 384, 3000), &Device::Cpu)?;
    let w = Tensor::randn(0f32, 1., (384, 384, 3), &Device::Cpu)?;
    let res = inp.conv1d(&w, 0, 1);
    println!("{res:?}");
    let start = std::time::Instant::now();
    for i in 0..N_ITERS {
        let res = inp.conv1d(&w, 0, 1);
        println!("{i} {res:?}");
    }
    println!("{:?}", start.elapsed() / N_ITERS as u32);
    Ok(())
}
