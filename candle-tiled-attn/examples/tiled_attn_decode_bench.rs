use anyhow::Result;
use candle::{DType, Device, Tensor};
use clap::Parser;
use std::time::Instant;

#[derive(Clone, Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 7)]
    batch: usize,
    #[arg(long, default_value_t = 8)]
    heads: usize,
    #[arg(long, default_value_t = 128)]
    kv_len: usize,
    #[arg(long, default_value_t = 256)]
    head_dim: usize,
    #[arg(long, default_value_t = 1000)]
    iters: usize,
    #[arg(long, default_value_t = 20)]
    warmup: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::new_cuda(0)?;
    let scale = 1.0f32 / (args.head_dim as f32).sqrt();
    let shape_q = (args.batch, args.heads, 1, args.head_dim);
    let shape_kv = (args.batch, args.heads, args.kv_len, args.head_dim);
    let q = Tensor::randn(0f32, 0.25, shape_q, &device)?.to_dtype(DType::F16)?;
    let k = Tensor::randn(0f32, 0.25, shape_kv, &device)?.to_dtype(DType::F16)?;
    let v = Tensor::randn(0f32, 0.25, shape_kv, &device)?.to_dtype(DType::F16)?;

    for _ in 0..args.warmup {
        let _ = candle_tiled_attn::tiled_attn_decode(&q, &k, &v, scale)?;
    }
    device.synchronize()?;

    let start = Instant::now();
    for _ in 0..args.iters {
        let _ = candle_tiled_attn::tiled_attn_decode(&q, &k, &v, scale)?;
    }
    device.synchronize()?;

    let elapsed = start.elapsed();
    let us = elapsed.as_secs_f64() * 1e6 / args.iters as f64;
    println!(
        "B={} H={} K={} D={} iters={} time={us:.3} us/iter",
        args.batch, args.heads, args.kv_len, args.head_dim, args.iters
    );
    Ok(())
}
