//! TurboQuant KV Cache Benchmark
//!
//! Simulates transformer attention with realistic model configs and compares
//! plain (ConcatKvCache) vs TurboQuant-quantized (QuantizedKvCache) KV caches.

use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_nn::kv_cache::{ConcatKvCache, KvCacheOps, QuantizedKvCache};
use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "TurboQuant KV Cache Benchmark")]
struct Args {
    /// Sequence length to simulate.
    #[arg(long, default_value_t = 512)]
    seq_len: usize,

    /// Batch size.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Use CPU even when GPU is available.
    #[arg(long)]
    cpu: bool,
}

struct ModelConfig {
    name: &'static str,
    head_dim: usize,
    num_kv_heads: usize,
    num_layers: usize,
}

const MODELS: &[ModelConfig] = &[
    ModelConfig {
        name: "Llama-3.1-8B",
        head_dim: 128,
        num_kv_heads: 8,
        num_layers: 32,
    },
    ModelConfig {
        name: "Qwen3-4B",
        head_dim: 128,
        num_kv_heads: 8,
        num_layers: 36,
    },
];

const BIT_WIDTHS: &[usize] = &[2, 3, 4];
const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;
const CONCAT_DIM: usize = 2; // [B, H, S, D]

struct BenchResult {
    label: String,
    k_recon_mse: f64,
    attn_score_err: f64,
    time_ms: f64,
    mem_bytes_per_vec: usize,
}

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

/// Run a single configuration and return bench results.
fn bench_config(
    model: &ModelConfig,
    seq_len: usize,
    batch: usize,
    device: &Device,
) -> Result<Vec<BenchResult>> {
    let shape = (batch, model.num_kv_heads, seq_len, model.head_dim);

    // Generate random Q, K, V tensors.
    let q = Tensor::randn(0f32, 1f32, shape, device)?;
    let k = Tensor::randn(0f32, 1f32, shape, device)?;
    let v = Tensor::randn(0f32, 1f32, shape, device)?;

    // --- Plain (ConcatKvCache) baseline ---
    let plain_time = {
        let mut total = std::time::Duration::ZERO;
        // warmup
        for _ in 0..WARMUP_ITERS {
            let mut cache = ConcatKvCache::new(CONCAT_DIM);
            let _ = cache.append(&k, &v)?;
        }
        for _ in 0..BENCH_ITERS {
            let mut cache = ConcatKvCache::new(CONCAT_DIM);
            let start = Instant::now();
            let _ = cache.append(&k, &v)?;
            total += start.elapsed();
        }
        total.as_secs_f64() / BENCH_ITERS as f64 * 1000.0
    };

    let mut plain_cache = ConcatKvCache::new(CONCAT_DIM);
    let (plain_k, _plain_v) = plain_cache.append(&k, &v)?;

    // Attention logits: softmax(Q @ K^T / sqrt(d))
    let scale = (model.head_dim as f64).sqrt();
    let plain_logits = q.matmul(&plain_k.t()?)?.affine(1.0 / scale, 0.0)?;

    let plain_mem = model.head_dim * 4; // F32 = 4 bytes per element

    let mut results = vec![BenchResult {
        label: "plain".to_string(),
        k_recon_mse: 0.0,
        attn_score_err: 0.0,
        time_ms: plain_time,
        mem_bytes_per_vec: plain_mem,
    }];

    // --- Quantized (QuantizedKvCache) for each bit width ---
    for &bits in BIT_WIDTHS {
        // Timing
        let quant_time = {
            let mut total = std::time::Duration::ZERO;
            for i in 0..(WARMUP_ITERS + BENCH_ITERS) {
                let mut cache =
                    QuantizedKvCache::new(CONCAT_DIM, model.head_dim, bits, DType::F32, device)?;
                let start = Instant::now();
                let _ = cache.append(&k, &v)?;
                let elapsed = start.elapsed();
                if i >= WARMUP_ITERS {
                    total += elapsed;
                }
            }
            total.as_secs_f64() / BENCH_ITERS as f64 * 1000.0
        };

        // Quality metrics
        let mut cache =
            QuantizedKvCache::new(CONCAT_DIM, model.head_dim, bits, DType::F32, device)?;
        let (quant_k, quant_v) = cache.append(&k, &v)?;
        let _ = &quant_v; // used for cache completeness

        let k_mse = relative_mse(&quant_k, &k)?;

        let quant_logits = q.matmul(&quant_k.t()?)?.affine(1.0 / scale, 0.0)?;
        let attn_err = relative_mse(&quant_logits, &plain_logits)?;

        // Memory: U8 indices (head_dim bytes) + F32 norm (4 bytes)
        let quant_mem = model.head_dim + 4;

        results.push(BenchResult {
            label: format!("{}", bits),
            k_recon_mse: k_mse,
            attn_score_err: attn_err,
            time_ms: quant_time,
            mem_bytes_per_vec: quant_mem,
        });
    }

    Ok(results)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    println!("TurboQuant KV Cache Benchmark");
    println!("=============================\n");

    for model in MODELS {
        println!(
            "Model: {} (head_dim={}, kv_heads={}, layers={})",
            model.name, model.head_dim, model.num_kv_heads, model.num_layers
        );
        println!(
            "Sequence length: {}, Batch: {}\n",
            args.seq_len, args.batch_size
        );

        let results = bench_config(model, args.seq_len, args.batch_size, &device)?;

        println!(
            "  {:>5} | {:>11} | {:>14} | {:>15} | {:>18}",
            "Bits", "K Recon MSE", "Attn Score Err", "Quant Time (ms)", "Memory (bytes/vec)"
        );
        println!(
            "  {}|{}|{}|{}|{}",
            "-".repeat(6),
            "-".repeat(13),
            "-".repeat(16),
            "-".repeat(17),
            "-".repeat(20)
        );

        for r in &results {
            println!(
                "  {:>5} | {:>11.4} | {:>14.4} | {:>15.2} | {:>18}",
                r.label, r.k_recon_mse, r.attn_score_err, r.time_ms, r.mem_bytes_per_vec
            );
        }
        println!();
    }

    Ok(())
}
