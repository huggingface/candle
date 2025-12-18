use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Result, Tensor};
use criterion::{criterion_group, Criterion};
use std::hint::black_box;
use std::time::Instant;

use candle_nn::cpu_flash_attention::flash_attn_varlen_cpu_fast_f32;
use candle_nn::varlen_attention::flash_attn_varlen_unfused; // TODO: implement for every

// Use the shared helper (moved out of #[cfg(test)]).
use rand::prelude::*;

/// Same helper you already have (copied out of #[cfg(test)]).
/// Prefill-style varlen: seqlens_q == seqlens_k per batch element.
#[allow(clippy::type_complexity)]
pub fn make_varlen_inputs_prefill(
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, usize, usize)> {
    let mut rng = StdRng::seed_from_u64(456);

    let mut seqlens = Vec::<u32>::with_capacity(batch_size);
    let mut total = 0usize;
    let mut max_l = 0usize;

    for _ in 0..batch_size {
        let l = rng.random_range(4..=max_seq);
        seqlens.push(l as u32);
        total += l;
        max_l = max_l.max(l);
    }

    // Q: [total, Hq, D], K/V: [total, Hk, D]
    let q_data: Vec<f32> = (0..total * num_heads * head_dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let k_data: Vec<f32> = (0..total * num_kv_heads * head_dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let v_data: Vec<f32> = (0..total * num_kv_heads * head_dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    let q = Tensor::from_vec(q_data, (total, num_heads, head_dim), device)?;
    let k = Tensor::from_vec(k_data, (total, num_kv_heads, head_dim), device)?;
    let v = Tensor::from_vec(v_data, (total, num_kv_heads, head_dim), device)?;

    let seqlens_q = Tensor::from_vec(seqlens.clone(), batch_size, device)?;
    let seqlens_k = Tensor::from_vec(seqlens, batch_size, device)?;

    // max_q == max_k == max_l for prefill
    Ok((q, k, v, seqlens_q, seqlens_k, max_l, max_l))
}

pub fn make_alibi_slopes(num_heads: usize, device: &Device) -> Result<Tensor> {
    let slopes: Vec<f32> = (0..num_heads)
        .map(|i| 2.0f32.powi(-(i as i32 + 1)))
        .collect();
    Tensor::from_vec(slopes, num_heads, device)
}

#[inline(never)]
fn run_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi: Option<&Tensor>,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_q: usize,
    max_k: usize,
    softmax_scale: f64,
    causal: bool,
    wl: Option<usize>,
    wr: Option<usize>,
) -> Result<Tensor> {
    flash_attn_varlen_cpu_fast_f32(
        q,
        k,
        v,
        alibi,
        seqlens_q,
        seqlens_k,
        max_q,
        max_k,
        softmax_scale,
        causal,
        wl,
        wr,
    )
}

fn bench_case(
    c: &mut Criterion,
    device: &Device,
    name: String,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alibi: Option<Tensor>,
    seqlens_q: Tensor,
    seqlens_k: Tensor,
    max_q: usize,
    max_k: usize,
    head_dim: usize,
    causal: bool,
    wl: Option<usize>,
    wr: Option<usize>,
) {
    let softmax_scale = 1.0 / (head_dim as f64).sqrt();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let _out = run_varlen(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    alibi.as_ref(),
                    black_box(&seqlens_q),
                    black_box(&seqlens_k),
                    max_q,
                    max_k,
                    softmax_scale,
                    causal,
                    wl,
                    wr,
                )
                .unwrap();
                std::hint::black_box(_out);
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
            continue; // this kernel is CPU-only
        }

        // Keep dtype fixed to f32 because flash_attn_varlen_cpu_fast_f32 is f32-only.
        let dtype = DType::F32;

        // Prefill baseline
        {
            let (b, hq, hk, d, max_seq) = (4usize, 8usize, 8usize, 64usize, 256usize);
            let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) =
                make_varlen_inputs_prefill(b, hq, hk, d, max_seq, &dev).unwrap();

            bench_case(
                c,
                &dev,
                format!("fa_varlen_prefill_{dtype:?}_b{b}_hq{hq}_hk{hk}_d{d}_noncausal"),
                q,
                k,
                v,
                None,
                seqlens_q,
                seqlens_k,
                max_q,
                max_k,
                d,
                false,
                None,
                None,
            );
        }

        // Prefill causal
        {
            let (b, hq, hk, d, max_seq) = (4usize, 8usize, 8usize, 64usize, 256usize);
            let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) =
                make_varlen_inputs_prefill(b, hq, hk, d, max_seq, &dev).unwrap();

            bench_case(
                c,
                &dev,
                format!("fa_varlen_prefill_{dtype:?}_b{b}_hq{hq}_hk{hk}_d{d}_causal"),
                q,
                k,
                v,
                None,
                seqlens_q,
                seqlens_k,
                max_q,
                max_k,
                d,
                true,
                None,
                None,
            );
        }

        // Prefill sliding-window (Mistral-style: only left, causal=true, right=None)
        {
            let (b, hq, hk, d, max_seq) = (4usize, 8usize, 8usize, 64usize, 256usize);
            let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) =
                make_varlen_inputs_prefill(b, hq, hk, d, max_seq, &dev).unwrap();

            bench_case(
                c,
                &dev,
                format!("fa_varlen_prefill_{dtype:?}_b{b}_hq{hq}_hk{hk}_d{d}_causal_wl128"),
                q,
                k,
                v,
                None,
                seqlens_q,
                seqlens_k,
                max_q,
                max_k,
                d,
                true,
                Some(128),
                None,
            );
        }

        // Prefill GQA (hq > hk == hv)
        {
            let (b, hq, hk, d, max_seq) = (3usize, 12usize, 4usize, 64usize, 256usize);
            let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) =
                make_varlen_inputs_prefill(b, hq, hk, d, max_seq, &dev).unwrap();

            bench_case(
                c,
                &dev,
                format!("fa_varlen_prefill_{dtype:?}_b{b}_hq{hq}_hk{hk}_d{d}_gqa_noncausal"),
                q,
                k,
                v,
                None,
                seqlens_q,
                seqlens_k,
                max_q,
                max_k,
                d,
                false,
                None,
                None,
            );
        }

        // Prefill ALiBi (causal)
        {
            let (b, hq, hk, d, max_seq) = (2usize, 8usize, 8usize, 64usize, 256usize);
            let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) =
                make_varlen_inputs_prefill(b, hq, hk, d, max_seq, &dev).unwrap();
            let alibi = Some(make_alibi_slopes(hq, &dev).unwrap());

            bench_case(
                c,
                &dev,
                format!("fa_varlen_prefill_{dtype:?}_b{b}_hq{hq}_hk{hk}_d{d}_alibi_causal"),
                q,
                k,
                v,
                alibi,
                seqlens_q,
                seqlens_k,
                max_q,
                max_k,
                d,
                true,
                None,
                None,
            );
        }
    }
}

criterion_group!(benches, criterion_benchmark);
