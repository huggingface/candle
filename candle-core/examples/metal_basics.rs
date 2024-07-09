#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    // This requires the code to be run with MTL_CAPTURE_ENABLED=1
    if let Device::Metal(m) = &device {
        m.capture("basics.gputrace")?;
    };
    let x = Tensor::randn(0f32, 1.0, (8 * 4096, 8 * 4096), &device)?;
    let _x1 = x.add(&x)?;
    let _x1 = x.matmul(&x)?;
    Ok(())
}
