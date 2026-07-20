// Regression check for https://github.com/huggingface/candle/issues/2271
//
// Repeatedly multiplying a 784x100 matrix by a 100x10 matrix used to leak memory
// on the Metal backend: autoreleased command buffers / encoders accumulated in the
// thread's autorelease pool, so the resident set grew without bound (>5 GB reported,
// >100 GB for BERT-sized models). The CPU backend never leaked.
//
// With the fix the resident set stays flat on both backends.
//
// Run with:
//   cargo run --release --example matmul_leak --features metal
//   cargo run --release --example matmul_leak            # CPU baseline

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_core::{Device, Tensor};

// Read the current process resident set size (RSS) in bytes via ps.
fn rss_bytes() -> u64 {
    let pid = std::process::id();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .expect("failed to run ps");
    let s = String::from_utf8_lossy(&out.stdout);
    // ps reports RSS in kilobytes.
    s.trim().parse::<u64>().unwrap_or(0) * 1024
}

fn main() -> Result<()> {
    let device = if cfg!(feature = "metal") {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };
    println!("device: {device:?}");

    let a = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
    let b = Tensor::randn(0f32, 1.0, (100, 10), &device)?;

    let iterations = 200_000usize;
    let start_rss = rss_bytes();
    println!("start RSS: {:.2} MB", start_rss as f64 / 1e6);

    for i in 0..iterations {
        let c = a.matmul(&b)?;
        // Force the computation to actually run and read the result back to the CPU.
        let _ = c.sum_all()?.to_scalar::<f32>()?;

        if i % 5_000 == 0 {
            let rss = rss_bytes();
            println!(
                "iter {i:>7}  RSS {:>8.2} MB  (delta {:>8.2} MB)",
                rss as f64 / 1e6,
                (rss as i64 - start_rss as i64) as f64 / 1e6,
            );
        }
    }

    let end_rss = rss_bytes();
    println!(
        "end RSS: {:.2} MB  (grew {:.2} MB over {iterations} iterations)",
        end_rss as f64 / 1e6,
        (end_rss as i64 - start_rss as i64) as f64 / 1e6,
    );
    Ok(())
}
