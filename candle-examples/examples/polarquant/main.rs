//! PolarQuant Vector Quantization Demo
//!
//! Demonstrates standalone PolarQuant (arXiv:2502.02617) for MSE-optimal vector
//! compression. Quantizes random embedding vectors at different bit widths and
//! reports reconstruction quality, compression ratio, and throughput.

use anyhow::Result;
use candle::{DType, Tensor};
use candle_nn::polarquant::PolarQuant;
use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "PolarQuant Vector Quantization Demo")]
struct Args {
    /// Vector dimension.
    #[arg(long, default_value_t = 128)]
    dim: usize,

    /// Number of vectors to quantize.
    #[arg(long, default_value_t = 1024)]
    num_vectors: usize,

    /// Use CPU even when GPU is available.
    #[arg(long)]
    cpu: bool,
}

const BIT_WIDTHS: &[usize] = &[1, 2, 3, 4];

/// Compute relative MSE: ||a - b||² / ||b||²
fn relative_mse(a: &Tensor, b: &Tensor) -> Result<f64> {
    let diff = a.sub(b)?;
    let num = diff
        .sqr()?
        .sum_all()?
        .to_dtype(DType::F64)?
        .to_scalar::<f64>()?;
    let den = b
        .sqr()?
        .sum_all()?
        .to_dtype(DType::F64)?
        .to_scalar::<f64>()?;
    Ok(if den == 0.0 { 0.0 } else { num / den })
}

/// Compute cosine similarity between corresponding rows, averaged.
fn avg_cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f64> {
    let dot = (a * b)?.sum(1)?.to_dtype(DType::F64)?;
    let norm_a = a.sqr()?.sum(1)?.sqrt()?.to_dtype(DType::F64)?;
    let norm_b = b.sqr()?.sum(1)?.sqrt()?.to_dtype(DType::F64)?;
    let cos = dot.div(&(norm_a * norm_b)?)?;
    let avg = cos.mean_all()?.to_scalar::<f64>()?;
    Ok(avg)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    println!("PolarQuant Vector Quantization Demo");
    println!("===================================");
    println!(
        "Vectors: {} × dim={}, dtype=F32\n",
        args.num_vectors, args.dim
    );

    // Generate random vectors (simulating embeddings)
    let x = Tensor::randn(0f32, 1f32, (args.num_vectors, args.dim), &device)?;

    // F32 baseline memory
    let f32_bytes_per_vec = args.dim * 4;

    println!(
        "  {:>4} | {:>12} | {:>10} | {:>12} | {:>15} | {:>12}",
        "Bits", "Relative MSE", "Cosine Sim", "Compression", "Quant+Deq (ms)", "Vecs/sec"
    );
    println!(
        "  {}|{}|{}|{}|{}|{}",
        "-".repeat(5),
        "-".repeat(14),
        "-".repeat(12),
        "-".repeat(14),
        "-".repeat(17),
        "-".repeat(14)
    );

    for &bits in BIT_WIDTHS {
        let quantizer = PolarQuant::new(args.dim, bits, DType::F32, &device)?;

        // Warmup
        for _ in 0..3 {
            let q = quantizer.quantize(&x)?;
            let _ = quantizer.dequantize(&q)?;
        }

        // Timed run
        let start = Instant::now();
        let q = quantizer.quantize(&x)?;
        let x_recon = quantizer.dequantize(&q)?;
        let elapsed = start.elapsed();

        let mse = relative_mse(&x_recon, &x)?;
        let cosine = avg_cosine_similarity(&x_recon, &x)?;

        // Compressed size: b bits per coordinate -> ceil(b*d/8) bytes + 4 bytes for norm
        let quant_bytes_per_vec = (bits * args.dim + 7) / 8 + 4;
        let compression = f32_bytes_per_vec as f64 / quant_bytes_per_vec as f64;

        let time_ms = elapsed.as_secs_f64() * 1000.0;
        let vecs_per_sec = args.num_vectors as f64 / elapsed.as_secs_f64();

        println!(
            "  {:>4} | {:>12.6} | {:>10.6} | {:>11.1}x | {:>15.2} | {:>12.0}",
            bits, mse, cosine, compression, time_ms, vecs_per_sec
        );
    }

    println!();
    println!("Notes:");
    println!("  - Relative MSE = ||x - x'||^2 / ||x||^2");
    println!("  - Cosine similarity measures directional preservation (1.0 = perfect)");
    println!(
        "  - Compression ratio relative to F32 ({} bytes/vec)",
        f32_bytes_per_vec
    );
    println!("  - PolarQuant is data-oblivious: no training or calibration needed");

    Ok(())
}
