//! TurboQuant KV Cache Benchmark (Synthetic Model)
//!
//! Builds a synthetic multi-layer transformer MLP with F32 weights,
//! then benchmarks KV cache compression using TurboQuantMse-quantized cache
//! vs plain concatenation cache.

use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_nn::kv_cache::{ConcatKvCache, QuantizedKvCache};
use candle_nn::{Linear, Module};
use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "TurboQuant KV Cache Benchmark (Synthetic Model)")]
struct Args {
    /// Use CPU even when GPU is available.
    #[arg(long)]
    cpu: bool,

    /// Quantization bit width for KV cache.
    #[arg(long, default_value_t = 4)]
    bits: usize,

    /// Number of transformer layers.
    #[arg(long, default_value_t = 12)]
    num_layers: usize,

    /// Hidden dimension.
    #[arg(long, default_value_t = 768)]
    hidden_dim: usize,

    /// Intermediate (FFN) dimension.
    #[arg(long, default_value_t = 3072)]
    intermediate_dim: usize,

    /// Number of attention heads.
    #[arg(long, default_value_t = 12)]
    num_heads: usize,

    /// Batch size.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Sequence length.
    #[arg(long, default_value_t = 128)]
    seq_len: usize,

    /// Number of inference iterations for timing.
    #[arg(long, default_value_t = 5)]
    iters: usize,
}

/// A single transformer layer's worth of Linear weights (attention + MLP).
struct TransformerLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl TransformerLayer {
    fn new(hidden: usize, intermediate: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            q_proj: Linear::new(Tensor::randn(0f32, 0.02, (hidden, hidden), device)?, None),
            k_proj: Linear::new(Tensor::randn(0f32, 0.02, (hidden, hidden), device)?, None),
            v_proj: Linear::new(Tensor::randn(0f32, 0.02, (hidden, hidden), device)?, None),
            o_proj: Linear::new(Tensor::randn(0f32, 0.02, (hidden, hidden), device)?, None),
            gate_proj: Linear::new(
                Tensor::randn(0f32, 0.02, (intermediate, hidden), device)?,
                None,
            ),
            up_proj: Linear::new(
                Tensor::randn(0f32, 0.02, (intermediate, hidden), device)?,
                None,
            ),
            down_proj: Linear::new(
                Tensor::randn(0f32, 0.02, (hidden, intermediate), device)?,
                None,
            ),
        })
    }

    fn forward_mlp(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = (candle_nn::Activation::Silu.forward(&gate)? * up)?;
        let down = self.down_proj.forward(&activated)?;
        Ok(down)
    }

    fn weight_bytes(&self) -> usize {
        fn tensor_bytes(t: &Tensor) -> usize {
            t.elem_count() * 4
        }
        tensor_bytes(self.q_proj.weight())
            + tensor_bytes(self.k_proj.weight())
            + tensor_bytes(self.v_proj.weight())
            + tensor_bytes(self.o_proj.weight())
            + tensor_bytes(self.gate_proj.weight())
            + tensor_bytes(self.up_proj.weight())
            + tensor_bytes(self.down_proj.weight())
    }
}

fn bytes_to_mb(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn relative_mse(a: &Tensor, b: &Tensor) -> Result<f64> {
    let diff = (a - b)?;
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

fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f64> {
    let a_flat = a.flatten_all()?.to_dtype(DType::F64)?;
    let b_flat = b.flatten_all()?.to_dtype(DType::F64)?;
    let dot = (&a_flat * &b_flat)?.sum_all()?.to_scalar::<f64>()?;
    let na = a_flat.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt();
    let nb = b_flat.sqr()?.sum_all()?.to_scalar::<f64>()?.sqrt();
    Ok(dot / (na * nb + 1e-10))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    println!("TurboQuant KV Cache Benchmark (Synthetic Model)");
    println!("================================================");
    println!(
        "Config: {}L, hidden={}, intermediate={}, heads={}",
        args.num_layers, args.hidden_dim, args.intermediate_dim, args.num_heads
    );
    println!("KV cache quantization: {}-bit TurboQuantMse", args.bits);
    println!(
        "Inference: batch={}, seq_len={}, iters={}\n",
        args.batch_size, args.seq_len, args.iters
    );

    // ── Phase 1: Build F32 model ──────────────────────────────────────
    print!("Building F32 model... ");
    let start = Instant::now();
    let layers: Vec<TransformerLayer> = (0..args.num_layers)
        .map(|_| TransformerLayer::new(args.hidden_dim, args.intermediate_dim, &device))
        .collect::<Result<_>>()?;
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    let weight_bytes: usize = layers.iter().map(|l| l.weight_bytes()).sum();
    println!(
        "done ({build_ms:.0}ms, {:.1} MB)",
        bytes_to_mb(weight_bytes)
    );

    // ── Phase 2: MLP throughput ──────────────────────────────────────
    let x = Tensor::randn(
        0f32,
        1f32,
        (args.batch_size, args.seq_len, args.hidden_dim),
        &device,
    )?;

    println!(
        "\n── F32 MLP Throughput ({} iters) ────────────────────",
        args.iters
    );
    // Warmup
    for layer in &layers {
        let _ = layer.forward_mlp(&x)?;
    }
    let start = Instant::now();
    for _ in 0..args.iters {
        for layer in &layers {
            let _ = layer.forward_mlp(&x)?;
        }
    }
    let elapsed = start.elapsed();
    let tokens_per_sec =
        (args.iters * args.batch_size * args.seq_len) as f64 / elapsed.as_secs_f64();
    println!(
        "  {:.0} tokens/sec ({:.1}ms/iter)",
        tokens_per_sec,
        elapsed.as_secs_f64() * 1000.0 / args.iters as f64
    );

    // ── Phase 3: KV Cache compression ────────────────────────────────
    println!("\n── KV Cache Compression ────────────────────────────");
    let head_dim = args.hidden_dim / args.num_heads;
    let kv_shape = (args.batch_size, args.num_heads, args.seq_len, head_dim);
    let k = Tensor::randn(0f32, 1f32, kv_shape, &device)?;
    let v = Tensor::randn(0f32, 1f32, kv_shape, &device)?;

    // Plain cache
    let mut plain_cache = ConcatKvCache::new(2);
    let (plain_k, _) = plain_cache.append(&k, &v)?;

    // Quantized cache
    let mut quant_cache = QuantizedKvCache::new(2, head_dim, args.bits, DType::F32, &device)?;
    let (quant_k, _) = quant_cache.append(&k, &v)?;

    let kv_mse = relative_mse(&quant_k, &plain_k)?;
    let kv_cos = cosine_similarity(&quant_k, &plain_k)?;

    let plain_kv_bytes = args.batch_size * args.num_heads * args.seq_len * head_dim * 4 * 2;
    let quant_kv_bytes = args.batch_size * args.num_heads * args.seq_len * (head_dim + 4) * 2;

    println!(
        "  Plain KV:    {:.1} MB ({} positions)",
        bytes_to_mb(plain_kv_bytes),
        args.seq_len
    );
    println!(
        "  Quant-{} KV: {:.1} MB ({:.1}x compression)",
        args.bits,
        bytes_to_mb(quant_kv_bytes),
        plain_kv_bytes as f64 / quant_kv_bytes as f64
    );
    println!("  K recon:     MSE={kv_mse:.6}, cos={kv_cos:.6}");

    // ── Summary ──────────────────────────────────────────────────────
    let total_plain = weight_bytes + plain_kv_bytes;
    let total_quant = weight_bytes + quant_kv_bytes;
    println!("\n── Total Memory (F32 weights + KV cache) ───────────");
    println!("  Plain KV:    {:.1} MB", bytes_to_mb(total_plain));
    println!(
        "  Quant-{} KV: {:.1} MB ({:.1}x KV compression, {:.0}% total savings)",
        args.bits,
        bytes_to_mb(total_quant),
        plain_kv_bytes as f64 / quant_kv_bytes as f64,
        (1.0 - total_quant as f64 / total_plain as f64) * 100.0
    );

    Ok(())
}
