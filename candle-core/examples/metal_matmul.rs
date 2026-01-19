#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    println!("Metal Matmul Example\n");

    // Test various matmul shapes
    let test_cases = vec![
        ("Small 2D", vec![256, 256], vec![256, 256]),
        ("Large 2D", vec![1024, 1024], vec![1024, 1024]),
        ("3D Batch", vec![32, 128, 64], vec![32, 64, 128]),
        ("4D Attention", vec![4, 8, 64, 32], vec![4, 8, 32, 64]),
    ];

    for (name, shape_a, shape_b) in test_cases {
        println!("Test: {}", name);
        println!("  Shape A: {:?}, Shape B: {:?}", shape_a, shape_b);

        let a = Tensor::randn(0f32, 1.0, shape_a.as_slice(), &device)?;
        let b = Tensor::randn(0f32, 1.0, shape_b.as_slice(), &device)?;
        let c = a.matmul(&b)?;

        println!("  Output shape: {:?}", c.shape());
        println!();
    }

    Ok(())
}
