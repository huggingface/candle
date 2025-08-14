#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let a = Tensor::new(&[[0.0f32, 1.0], [2.0, 3.0], [4.0, 5.0]], &Device::Cpu)?;
    let b = Tensor::new(&[[66.0f32, 77.0], [88.0f32, 99.0]], &Device::Cpu)?;
    let new_a = a.slice_scatter(&b, 0, 1)?;
    assert_eq!(a.to_vec2::<f32>()?, [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]);
    assert_eq!(
        new_a.to_vec2::<f32>()?,
        [[0.0, 1.0], [66.0, 77.0], [88.0, 99.0]]
    );
    Ok(())
}
