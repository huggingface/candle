//! Metal GEMM Benchmark
//!
//! This benchmark compares Candle's Metal GEMM performance against MLX.
//! Run with: cargo run --example metal_matmul_benchmark --features metal --release

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{Device, Tensor};
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

    // Warmup with sync
    for _ in 0..warmup {
        let _c = a.broadcast_matmul(&b)?;
        device.synchronize()?;
    }

    // Benchmark with proper sync using device.synchronize()
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let _c = a.broadcast_matmul(&b)?;
        device.synchronize()?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    Ok(times.iter().sum::<f64>() / times.len() as f64)
}

fn calculate_gflops(shape_a: &[usize], shape_b: &[usize], ms: f64) -> f64 {
    let batch: usize = shape_a
        .iter()
        .take(shape_a.len().saturating_sub(2))
        .product();
    let batch = if batch == 0 { 1 } else { batch };
    let m = shape_a[shape_a.len() - 2];
    let k = shape_a[shape_a.len() - 1];
    let n = shape_b[shape_b.len() - 1];
    let flops = 2.0 * batch as f64 * m as f64 * k as f64 * n as f64;
    flops / (ms / 1000.0) / 1e9
}

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    println!("Metal GEMM Benchmark (Candle)\n");

    // Test cases covering common GEMM scenarios
    let test_cases: Vec<(&str, Vec<usize>, Vec<usize>)> = vec![
        // 4D Attention scenarios (multi-head attention)
        (
            "4D Attn Small",
            vec![484, 6, 144, 32],
            vec![484, 6, 32, 144],
        ),
        (
            "4D Attn Large",
            vec![121, 24, 144, 32],
            vec![121, 24, 32, 144],
        ),
        // 3D QKV projection scenarios
        ("3D QKV Small", vec![484, 144, 192], vec![484, 192, 576]),
        ("3D QKV Large", vec![121, 144, 768], vec![121, 768, 2304]),
        // Square matrix tests (common benchmark)
        ("Square 256", vec![256, 256], vec![256, 256]),
        ("Square 512", vec![512, 512], vec![512, 512]),
        ("Square 1024", vec![1024, 1024], vec![1024, 1024]),
        ("Square 2048", vec![2048, 2048], vec![2048, 2048]),
        // 2D Linear layer scenarios (transformer FFN)
        ("2D Linear Small", vec![69696, 192], vec![192, 768]),
        ("2D Linear Large", vec![17424, 768], vec![768, 3072]),
        // 3D Batch matmul (attention patterns)
        ("Batch 100", vec![100, 144, 32], vec![100, 32, 144]),
        ("Batch 1000", vec![1000, 144, 32], vec![1000, 32, 144]),
        ("Batch 2904", vec![2904, 144, 32], vec![2904, 32, 144]),
    ];

    println!(
        "{:<20} {:<25} {:<25} {:>10} {:>12}",
        "Test Case", "Shape A", "Shape B", "Time", "GFLOPS"
    );
    println!("{}", "-".repeat(95));

    for (name, shape_a, shape_b) in &test_cases {
        let ms = benchmark_matmul(shape_a, shape_b, &device, 5, 20)?;
        let gflops = calculate_gflops(shape_a, shape_b, ms);

        println!(
            "{:<20} {:<25} {:<25} {:>7.2}ms {:>7.1} GFLOPS",
            name,
            format!("{:?}", shape_a),
            format!("{:?}", shape_b),
            ms,
            gflops
        );
    }

    Ok(())
}
