#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn benchmark_matmul(
    shape_a: &[usize],
    shape_b: &[usize],
    device: &Device,
    warmup: usize,
    iterations: usize,
) -> Result<f64> {
    let a = Tensor::randn(0f32, 1.0, shape_a, device)?;
    let b = Tensor::randn(0f32, 1.0, shape_b, device)?;

    // Warmup
    for _ in 0..warmup {
        let _ = a.matmul(&b)?;
        let _ = Tensor::zeros(1, DType::F32, device).and_then(|t| t.to_vec1::<f32>());
    }

    // Benchmark
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let c = a.matmul(&b)?;
        let _ = c.flatten_all()?.get(0)?.to_scalar::<f32>()?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    Ok(times.iter().sum::<f64>() / times.len() as f64)
}

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    println!("Metal GEMM Benchmark\n");

    let test_cases = vec![
        ("4D Attn Small", vec![484, 6, 144, 32], vec![484, 6, 32, 144]),
        ("4D Attn Large", vec![121, 24, 144, 32], vec![121, 24, 32, 144]),
        ("3D QKV Small", vec![484, 144, 192], vec![484, 192, 576]),
        ("3D QKV Large", vec![121, 144, 768], vec![121, 768, 2304]),
        ("2D Linear Small", vec![69696, 192], vec![192, 768]),
        ("2D Linear Large", vec![17424, 768], vec![768, 3072]),
        ("Square 256", vec![256, 256], vec![256, 256]),
        ("Square 512", vec![512, 512], vec![512, 512]),
        ("Square 1024", vec![1024, 1024], vec![1024, 1024]),
        ("Square 2048", vec![2048, 2048], vec![2048, 2048]),
    ];

    println!("{:<20} {:<25} {:<25} {:>10}", "Test Case", "Shape A", "Shape B", "Time");
    println!("{}", "-".repeat(80));

    for (name, shape_a, shape_b) in test_cases {
        let ms = benchmark_matmul(&shape_a, &shape_b, &device, 5, 20)?;
        println!(
            "{:<20} {:<25} {:<25} {:>7.2}ms",
            name,
            format!("{:?}", shape_a),
            format!("{:?}", shape_b),
            ms
        );
    }

    Ok(())
}
