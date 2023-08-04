#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let mut file = std::fs::File::open("ggml.bin")?;
    let data = candle_core::ggml::Content::read(&mut file, &Device::Cpu)?;
    let a = Tensor::randn(0f32, 1., (2, 3), &Device::Cpu)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &Device::Cpu)?;
    let c = a.matmul(&b)?;
    println!("{a} {b} {c}");

    let data = &[[3f32, 1., 4., 1., 5.], [2., 7., 1., 8., 2.]];
    let t1 = Tensor::new(data, &Device::Cpu)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 7., 1., 8., 2.]];
    let t2 = Tensor::new(data2, &Device::Cpu)?;
    assert_eq!(
        Tensor::cat(&[&t1.t()?, &t2.t()?], 1)?
            .t()?
            .to_vec2::<f32>()?,
        [
            [3.0, 1.0, 4.0, 1.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0]
        ]
    );
    Ok(())
}
