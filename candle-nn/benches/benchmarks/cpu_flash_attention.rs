use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Result, Tensor};
use criterion::{criterion_group, Criterion};
use std::hint::black_box;
use std::time::Instant;

use candle_nn::cpu_flash_attention::run_flash_attn_cpu;

use rand::prelude::*;

#[allow(clippy::type_complexity)]
pub fn make_attention_inputs(
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut rng = StdRng::seed_from_u64(123);

    match dtype {
        DType::F32 => {
            let q_data: Vec<f32> = (0..batch_size * seq_len * num_heads * head_dim)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let k_data: Vec<f32> = (0..batch_size * seq_len * num_kv_heads * head_dim)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let v_data: Vec<f32> = (0..batch_size * seq_len * num_kv_heads * head_dim)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();

            let q = Tensor::from_vec(q_data, (batch_size, seq_len, num_heads, head_dim), device)?;
            let k = Tensor::from_vec(
                k_data,
                (batch_size, seq_len, num_kv_heads, head_dim),
                device,
            )?;
            let v = Tensor::from_vec(
                v_data,
                (batch_size, seq_len, num_kv_heads, head_dim),
                device,
            )?;
            Ok((q, k, v))
        }
        DType::F16 => {
            let q_data: Vec<half::f16> = (0..batch_size * seq_len * num_heads * head_dim)
                .map(|_| half::f16::from_f32(rng.random_range(-1.0..1.0)))
                .collect();
            let k_data: Vec<half::f16> = (0..batch_size * seq_len * num_kv_heads * head_dim)
                .map(|_| half::f16::from_f32(rng.random_range(-1.0..1.0)))
                .collect();
            let v_data: Vec<half::f16> = (0..batch_size * seq_len * num_kv_heads * head_dim)
                .map(|_| half::f16::from_f32(rng.random_range(-1.0..1.0)))
                .collect();

            let q = Tensor::from_vec(q_data, (batch_size, seq_len, num_heads, head_dim), device)?;
            let k = Tensor::from_vec(
                k_data,
                (batch_size, seq_len, num_kv_heads, head_dim),
                device,
            )?;
            let v = Tensor::from_vec(
                v_data,
                (batch_size, seq_len, num_kv_heads, head_dim),
                device,
            )?;
            Ok((q, k, v))
        }
        _ => candle::bail!("Unsupported dtype for benchmark: {:?}", dtype),
    }
}

pub fn make_causal_mask(
    batch_size: usize,
    seq_len: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    match dtype {
        DType::F32 => {
            let mut mask_data = vec![0.0f32; batch_size * seq_len * seq_len];
            for b in 0..batch_size {
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        mask_data[(b * seq_len + i) * seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            Tensor::from_vec(mask_data, (batch_size, seq_len, seq_len), device)
        }
        DType::F16 => {
            let mut mask_data = vec![half::f16::from_f32(0.0); batch_size * seq_len * seq_len];
            for b in 0..batch_size {
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        mask_data[(b * seq_len + i) * seq_len + j] =
                            half::f16::from_f32(f32::NEG_INFINITY);
                    }
                }
            }
            Tensor::from_vec(mask_data, (batch_size, seq_len, seq_len), device)
        }
        _ => candle::bail!("Unsupported dtype for mask: {:?}", dtype),
    }
}

#[inline(never)]
fn run_flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    softmax_scale: f32,
    max_bias: Option<f32>,
    softcap: Option<f32>,
    dtype: DType,
) -> Result<Tensor> {
    match dtype {
        DType::F32 => run_flash_attn_cpu::<f32>(q, k, v, mask, softmax_scale, max_bias, softcap),
        DType::F16 => {
            run_flash_attn_cpu::<half::f16>(q, k, v, mask, softmax_scale, max_bias, softcap)
        }
        _ => candle::bail!(
            "Unsupported dtype for flash attention benchmark: {:?}",
            dtype
        ),
    }
}

fn bench_case(
    c: &mut Criterion,
    device: &Device,
    name: String,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Option<Tensor>,
    head_dim: usize,
    max_bias: Option<f32>,
    softcap: Option<f32>,
    dtype: DType,
) {
    let softmax_scale = 1.0 / (head_dim as f64).sqrt() as f32;

    let mut group = c.benchmark_group(device.bench_name(name));

    let q = q.clone();
    let k = k.clone();
    let v = v.clone();
    let mask = mask.clone();

    group.bench_function("cpu_flash_attention", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let out = run_flash_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    mask.as_ref(),
                    softmax_scale,
                    max_bias,
                    softcap,
                    dtype,
                )
                .unwrap();
                std::hint::black_box(out);
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for dev in handler.devices {
        if !dev.is_cpu() {
            continue;
        }

        for dtype in [DType::F16, DType::F32] {
            // Prefill baseline (comparable to varlen)
            {
                let (b, hq, hk, d, seq_len) = (4usize, 8usize, 8usize, 64usize, 256usize);
                let (q, k, v) = make_attention_inputs(b, seq_len, hq, hk, d, &dev, dtype).unwrap();

                bench_case(
                    c,
                    &dev,
                    format!("flash_prefill_{dtype:?}_b{b}_hq{hq}_hk{hk}_d{d}_noncausal"),
                    q,
                    k,
                    v,
                    None,
                    d,
                    None,
                    None,
                    dtype,
                );
            }

            // Prefill causal (comparable to varlen)
            {
                let (b, hq, hk, d, seq_len) = (4usize, 8usize, 8usize, 64usize, 256usize);
                let (q, k, v) = make_attention_inputs(b, seq_len, hq, hk, d, &dev, dtype).unwrap();
                let mask = Some(make_causal_mask(b, seq_len, &dev, dtype).unwrap());

                bench_case(
                    c,
                    &dev,
                    format!("flash_prefill_{dtype:?}_b{b}_hq{hq}_hk{hk}_d{d}_causal"),
                    q,
                    k,
                    v,
                    mask,
                    d,
                    None,
                    None,
                    dtype,
                );
            }

            // Prefill sliding-window (comparable to varlen)
            {
                let (b, hq, hk, d, seq_len) = (4usize, 8usize, 8usize, 64usize, 256usize);
                let (q, k, v) = make_attention_inputs(b, seq_len, hq, hk, d, &dev, dtype).unwrap();
                let mask = Some(make_causal_mask(b, seq_len, &dev, dtype).unwrap());

                bench_case(
                    c,
                    &dev,
                    format!("flash_prefill_{dtype:?}_b{b}_hq{hq}_hk{hk}_d{d}_causal_wl128"),
                    q,
                    k,
                    v,
                    mask,
                    d,
                    None,
                    None,
                    dtype,
                );
            }

            // Prefill GQA (comparable to varlen)
            {
                let (b, hq, hk, d, seq_len) = (3usize, 12usize, 4usize, 64usize, 256usize);
                let (q, k, v) = make_attention_inputs(b, seq_len, hq, hk, d, &dev, dtype).unwrap();

                bench_case(
                    c,
                    &dev,
                    format!("flash_prefill_{dtype:?}_b{b}_hq{hq}_hk{hk}_d{d}_gqa_noncausal"),
                    q,
                    k,
                    v,
                    None,
                    d,
                    None,
                    None,
                    dtype,
                );
            }

            // Prefill ALiBi (comparable to varlen)
            {
                let (b, hq, hk, d, seq_len) = (2usize, 8usize, 8usize, 64usize, 256usize);
                let (q, k, v) = make_attention_inputs(b, seq_len, hq, hk, d, &dev, dtype).unwrap();

                bench_case(
                    c,
                    &dev,
                    format!("flash_prefill_{dtype:?}_b{b}_hq{hq}_hk{hk}_d{d}_alibi_causal"),
                    q,
                    k,
                    v,
                    None,
                    d,
                    Some(8.0), // max_bias
                    None,
                    dtype,
                );
            }

            // Decode scenario (single query - flash attention optimization)
            {
                let (b, hq, hk, d, seq_len) = (4usize, 1usize, 32usize, 32usize, 128usize);
                let (q, k, v) = make_attention_inputs(b, seq_len, hq, hk, d, &dev, dtype).unwrap();

                bench_case(
                    c,
                    &dev,
                    format!("flash_decode_{dtype:?}_b{b}_hq{hq}_hk{hk}_d{d}_single"),
                    q,
                    k,
                    v,
                    None,
                    d,
                    None,
                    None,
                    dtype,
                );
            }

            // Large prefill (stress test)
            {
                let (b, hq, hk, d, seq_len) = (1usize, 8usize, 8usize, 64usize, 1024usize);
                let (q, k, v) = make_attention_inputs(b, seq_len, hq, hk, d, &dev, dtype).unwrap();

                bench_case(
                    c,
                    &dev,
                    format!("flash_prefill_{dtype:?}_b{b}_hq{hq}_hk{hk}_d{d}_large"),
                    q,
                    k,
                    v,
                    None,
                    d,
                    None,
                    None,
                    dtype,
                );
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
