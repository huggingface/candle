//! TurboQuantMse End-to-End Model Compression Benchmark
//!
//! Builds a realistic multi-layer transformer MLP, compresses all Linear layers
//! with TurboQuantMse, then benchmarks:
//! - Model size reduction (F32 vs quantized weights)
//! - Output quality (relative MSE, cosine similarity vs full-precision)
//! - Inference throughput (tokens/sec for full-precision vs quantized)
//! - KV cache compression (plain vs TurboQuantMse-quantized cache)

use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_nn::kv_cache::{ConcatKvCache, QuantizedKvCache};
use candle_nn::turboquant_nn::TurboQuantLinear;
use candle_nn::{Linear, Module};
use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "TurboQuantMse End-to-End Model Compression Benchmark")]
struct Args {
    /// Use CPU even when GPU is available.
    #[arg(long)]
    cpu: bool,

    /// Quantization bit width.
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
struct TransformerLayerF32 {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

struct TransformerLayerPQ {
    q_proj: TurboQuantLinear,
    k_proj: TurboQuantLinear,
    v_proj: TurboQuantLinear,
    o_proj: TurboQuantLinear,
    gate_proj: TurboQuantLinear,
    up_proj: TurboQuantLinear,
    down_proj: TurboQuantLinear,
}

impl TransformerLayerF32 {
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

    fn _forward_attn_projections(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        Ok((q, k, v))
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

impl TransformerLayerPQ {
    fn from_f32(layer: &TransformerLayerF32, bits: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            q_proj: TurboQuantLinear::from_linear(&layer.q_proj, bits, device)?,
            k_proj: TurboQuantLinear::from_linear(&layer.k_proj, bits, device)?,
            v_proj: TurboQuantLinear::from_linear(&layer.v_proj, bits, device)?,
            o_proj: TurboQuantLinear::from_linear(&layer.o_proj, bits, device)?,
            gate_proj: TurboQuantLinear::from_linear(&layer.gate_proj, bits, device)?,
            up_proj: TurboQuantLinear::from_linear(&layer.up_proj, bits, device)?,
            down_proj: TurboQuantLinear::from_linear(&layer.down_proj, bits, device)?,
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
        self.q_proj.quantized_weight_bytes()
            + self.k_proj.quantized_weight_bytes()
            + self.v_proj.quantized_weight_bytes()
            + self.o_proj.quantized_weight_bytes()
            + self.gate_proj.quantized_weight_bytes()
            + self.up_proj.quantized_weight_bytes()
            + self.down_proj.quantized_weight_bytes()
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

    println!("TurboQuantMse End-to-End Model Compression Benchmark");
    println!("==================================================");
    println!(
        "Config: {}L, hidden={}, intermediate={}, heads={}",
        args.num_layers, args.hidden_dim, args.intermediate_dim, args.num_heads
    );
    println!("Quantization: {}-bit TurboQuantMse", args.bits);
    println!(
        "Inference: batch={}, seq_len={}, iters={}\n",
        args.batch_size, args.seq_len, args.iters
    );

    // ── Phase 1: Build F32 model ──────────────────────────────────────
    print!("Building F32 model... ");
    let start = Instant::now();
    let f32_layers: Vec<TransformerLayerF32> = (0..args.num_layers)
        .map(|_| TransformerLayerF32::new(args.hidden_dim, args.intermediate_dim, &device))
        .collect::<Result<_>>()?;
    let f32_build_ms = start.elapsed().as_secs_f64() * 1000.0;
    let f32_bytes: usize = f32_layers.iter().map(|l| l.weight_bytes()).sum();
    println!(
        "done ({f32_build_ms:.0}ms, {:.1} MB)",
        bytes_to_mb(f32_bytes)
    );

    // ── Phase 2: Quantize to TurboQuantMse ───────────────────────────────
    print!("Quantizing to {}-bit TurboQuantMse... ", args.bits);
    let start = Instant::now();
    let pq_layers: Vec<TransformerLayerPQ> = f32_layers
        .iter()
        .map(|l| TransformerLayerPQ::from_f32(l, args.bits, &device))
        .collect::<Result<_>>()?;
    let quant_ms = start.elapsed().as_secs_f64() * 1000.0;
    let pq_bytes: usize = pq_layers.iter().map(|l| l.weight_bytes()).sum();
    println!("done ({quant_ms:.0}ms, {:.1} MB)", bytes_to_mb(pq_bytes));

    println!("\n── Model Size ──────────────────────────────────────");
    println!("  F32 weights:      {:>8.1} MB", bytes_to_mb(f32_bytes));
    println!(
        "  PQ-{} weights:    {:>8.1} MB",
        args.bits,
        bytes_to_mb(pq_bytes)
    );
    println!(
        "  Compression:      {:>8.1}x",
        f32_bytes as f64 / pq_bytes as f64
    );
    println!(
        "  Savings:          {:>8.1} MB ({:.0}%)",
        bytes_to_mb(f32_bytes - pq_bytes),
        (1.0 - pq_bytes as f64 / f32_bytes as f64) * 100.0
    );

    // ── Phase 3: Output quality ───────────────────────────────────────
    println!("\n── Output Quality (per layer, MLP path) ────────────");
    let x = Tensor::randn(
        0f32,
        1f32,
        (args.batch_size, args.seq_len, args.hidden_dim),
        &device,
    )?;

    let mut total_mse = 0.0;
    let mut total_cos = 0.0;
    for (i, (f32_layer, pq_layer)) in f32_layers.iter().zip(&pq_layers).enumerate() {
        let y_f32 = f32_layer.forward_mlp(&x)?;
        let y_pq = pq_layer.forward_mlp(&x)?;
        let mse = relative_mse(&y_pq, &y_f32)?;
        let cos = cosine_similarity(&y_pq, &y_f32)?;
        total_mse += mse;
        total_cos += cos;
        if i < 3 || i == args.num_layers - 1 {
            println!("  Layer {i:>2}: MSE={mse:.6}, cos={cos:.6}");
        } else if i == 3 {
            println!("  ...");
        }
    }
    println!(
        "  Average:  MSE={:.6}, cos={:.6}",
        total_mse / args.num_layers as f64,
        total_cos / args.num_layers as f64
    );

    // ── Phase 4: Inference throughput ─────────────────────────────────
    println!(
        "\n── Inference Throughput (MLP path, {} iters) ────────",
        args.iters
    );

    // Warmup
    for layer in &f32_layers {
        let _ = layer.forward_mlp(&x)?;
    }
    for layer in &pq_layers {
        let _ = layer.forward_mlp(&x)?;
    }

    // F32 timing
    let start = Instant::now();
    for _ in 0..args.iters {
        for layer in &f32_layers {
            let _ = layer.forward_mlp(&x)?;
        }
    }
    let f32_elapsed = start.elapsed();
    let f32_tokens_per_sec =
        (args.iters * args.batch_size * args.seq_len) as f64 / f32_elapsed.as_secs_f64();

    // PQ timing
    let start = Instant::now();
    for _ in 0..args.iters {
        for layer in &pq_layers {
            let _ = layer.forward_mlp(&x)?;
        }
    }
    let pq_elapsed = start.elapsed();
    let pq_tokens_per_sec =
        (args.iters * args.batch_size * args.seq_len) as f64 / pq_elapsed.as_secs_f64();

    println!(
        "  F32:  {:.0} tokens/sec ({:.1}ms/iter)",
        f32_tokens_per_sec,
        f32_elapsed.as_secs_f64() * 1000.0 / args.iters as f64
    );
    println!(
        "  PQ-{}: {:.0} tokens/sec ({:.1}ms/iter)",
        args.bits,
        pq_tokens_per_sec,
        pq_elapsed.as_secs_f64() * 1000.0 / args.iters as f64
    );
    println!(
        "  Overhead: {:.1}x slowdown (dequantize-on-the-fly)",
        f32_tokens_per_sec / pq_tokens_per_sec
    );

    // ── Phase 5: KV Cache compression ────────────────────────────────
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

    let plain_kv_bytes = args.batch_size * args.num_heads * args.seq_len * head_dim * 4 * 2; // K+V
    let quant_kv_bytes = args.batch_size * args.num_heads * args.seq_len * (head_dim + 4) * 2;

    println!(
        "  Plain KV:    {:.1} MB ({} positions)",
        bytes_to_mb(plain_kv_bytes),
        args.seq_len
    );
    println!(
        "  PQ-{} KV:   {:.1} MB ({:.1}x compression)",
        args.bits,
        bytes_to_mb(quant_kv_bytes),
        plain_kv_bytes as f64 / quant_kv_bytes as f64
    );
    println!("  K recon:     MSE={kv_mse:.6}, cos={kv_cos:.6}");

    // ── Summary ──────────────────────────────────────────────────────
    let total_f32 = f32_bytes + plain_kv_bytes;
    let total_pq = pq_bytes + quant_kv_bytes;
    println!("\n── Total Memory (weights + KV cache) ───────────────");
    println!("  F32:         {:.1} MB", bytes_to_mb(total_f32));
    println!(
        "  PQ-{}:       {:.1} MB ({:.1}x compression, {:.0}% savings)",
        args.bits,
        bytes_to_mb(total_pq),
        total_f32 as f64 / total_pq as f64,
        (1.0 - total_pq as f64 / total_f32 as f64) * 100.0
    );

    Ok(())
}
