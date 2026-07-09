//! TurboQuant Benchmark Example
//!
//! This example benchmarks the TurboQuant KV cache quantization algorithm
//! implemented in `candle_nn::quant_kv::turbo_quant`.
//!
//! - **TurboQuant**: Two-stage hybrid at 3.5 bits/channel (arxiv:2504.19874)
//!
//! ## Running
//!
//! ```bash
//! cargo run --example turboquant --release
//! ```

use candle::{DType, Device, Tensor};
use candle_nn::quant_kv::{
    prng::Prng,
    turbo_quant::{
        turbo_attention_scores, turbo_quantize, turbo_quantize_tensor, TurboQuantConfig,
    },
};

/// Generate a normalized random vector using our seeded PRNG.
fn random_unit_vec(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = Prng::new(seed);
    let mut v = vec![0.0f32; d];
    rng.fill_normal(&mut v);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

/// Generate a random f32 tensor of given shape.
fn random_tensor(shape: &[usize], seed: u64, device: &Device) -> Tensor {
    let total: usize = shape.iter().product();
    let mut rng = Prng::new(seed);
    let mut data = vec![0.0f32; total];
    rng.fill_normal(&mut data);
    let head_dim = *shape.last().unwrap();
    for chunk in data.chunks_mut(head_dim) {
        let norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in chunk.iter_mut() {
                *x /= norm;
            }
        }
    }
    Tensor::from_vec(data, shape, device).unwrap()
}

fn run_compression_benchmark() {
    println!("\n╔═════════════════════════════════════════════════════════╗");
    println!("║         Benchmark 1: Compression Ratios vs FP16         ║");
    println!("╠═════════════════════════════════════════════════════════╣");
    println!("║  Paper claims:                                          ║");
    println!("║    TurboQuant:  ~4.5× compression at 3.5 bits           ║");
    println!("╠══════════════╦══════════╦═══════════════════════════════╣");
    println!("║ head_dim     ║ FP16     ║ TurboQuant 3.5b               ║");
    println!("║              ║ (bytes)  ║ bits/dim  (compression)       ║");
    println!("╠══════════════╬══════════╬═══════════════════════════════╣");

    for &d in &[32usize, 64, 128, 256] {
        let fp16_bytes = d * 2;
        let k = random_unit_vec(d, 42);

        let turbo_cfg = TurboQuantConfig::new_3p5bit(d, 1);
        let turbo_key = turbo_quantize(&k, &turbo_cfg, 0);
        let turbo_bpd = turbo_key.bits_per_dim();

        println!(
            "║ d={d:<9} ║ {fp16_bytes:<8} ║ {turbo_bpd:>5.3} bits/d ({:.1}×)       ║",
            fp16_bytes as f32 * 8.0 / (turbo_key.byte_size() as f32 * 8.0),
        );
    }
    println!("╚══════════════╩══════════╩═══════════════════════════════╝");
}

fn run_accuracy_benchmark() {
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║             Benchmark 2: Inner Product Accuracy (N=2000 pairs)          ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Tests the unbiasedness guarantee: E[estimated ⟨q,k⟩] = true ⟨q,k⟩    ║");
    println!("╠═══════════════╦══════════════╦═══════════════╦══════════════╦════════════╣");
    println!("║ Algorithm     ║ Mean Error   ║ Mean Abs Err  ║ Std Dev Err  ║ 95th %ile  ║");
    println!("╠═══════════════╬══════════════╬═══════════════╬══════════════╬════════════╣");

    let d = 64usize;
    let n = 2000usize;

    let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..n)
        .map(|i| {
            (
                random_unit_vec(d, i as u64),
                random_unit_vec(d, i as u64 + 1_000_000),
            )
        })
        .collect();

    let true_dots: Vec<f32> = pairs
        .iter()
        .map(|(q, k)| q.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum())
        .collect();

    {
        let errors: Vec<f32> = pairs
            .iter()
            .zip(true_dots.iter())
            .map(|((q, k), &true_dot)| {
                let est: f32 = q.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum();
                est - true_dot
            })
            .collect();
        print_accuracy_row("FP16 (exact)", &errors);
    }

    {
        let cfg = TurboQuantConfig::new_3p5bit(d, 42);
        let errors: Vec<f32> = pairs
            .iter()
            .enumerate()
            .zip(true_dots.iter())
            .map(|((i, (q, k)), &true_dot)| {
                let tq = turbo_quantize(k, &cfg, i);
                let est = candle_nn::quant_kv::turbo_quant::turbo_inner_product(q, &tq, &cfg, i);
                est - true_dot
            })
            .collect();
        print_accuracy_row("TurboQ-3.5b", &errors);
    }

    {
        let cfg = TurboQuantConfig::new(d, 2.5, 42);
        let errors: Vec<f32> = pairs
            .iter()
            .enumerate()
            .zip(true_dots.iter())
            .map(|((i, (q, k)), &true_dot)| {
                let tq = turbo_quantize(k, &cfg, i);
                let est = candle_nn::quant_kv::turbo_quant::turbo_inner_product(q, &tq, &cfg, i);
                est - true_dot
            })
            .collect();
        print_accuracy_row("TurboQ-2.5b", &errors);
    }

    println!("╚═══════════════╩══════════════╩═══════════════╩══════════════╩════════════╝");
}

fn print_accuracy_row(name: &str, errors: &[f32]) {
    let n = errors.len() as f32;
    let mean = errors.iter().sum::<f32>() / n;
    let mean_abs = errors.iter().map(|e| e.abs()).sum::<f32>() / n;
    let variance = errors.iter().map(|e| (e - mean) * (e - mean)).sum::<f32>() / n;
    let std_dev = variance.sqrt();

    let mut sorted: Vec<f32> = errors.iter().map(|e| e.abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = sorted[(errors.len() as f32 * 0.95) as usize];

    println!("║ {name:<13} ║ {mean:>+12.4} ║ {mean_abs:>13.4} ║ {std_dev:>12.4} ║ {p95:>10.4} ║");
}

fn run_recall_benchmark(device: &Device) {
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║           Benchmark 3: Top-k Attention Recall (N=100 queries)           ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("╠═══════════════╦═══════════════╦═══════════════╦═══════════════╦════════╣");
    println!("║ Algorithm     ║ recall@1      ║ recall@5      ║ recall@10     ║ bits   ║");
    println!("╠═══════════════╬═══════════════╬═══════════════╬═══════════════╬════════╣");

    let d = 64usize;
    let n_tokens = 256usize;
    let n_queries = 100usize;
    let num_heads = 1usize;
    let seed_base = 12345u64;

    let k_tensor = random_tensor(&[1, num_heads, n_tokens, d], seed_base, device);

    let q_tensors: Vec<Tensor> = (0..n_queries)
        .map(|i| {
            random_tensor(
                &[1, num_heads, 1, d],
                seed_base + i as u64 + 100_000,
                device,
            )
        })
        .collect();

    fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.iter().take(k).map(|(i, _)| *i).collect()
    }

    fn recall_at_k(true_topk: &[usize], pred_topk: &[usize]) -> f32 {
        let true_set: std::collections::HashSet<usize> = true_topk.iter().copied().collect();
        let hits = pred_topk.iter().filter(|&&i| true_set.contains(&i)).count();
        hits as f32 / true_topk.len() as f32
    }

    let k_f32 = k_tensor.to_dtype(DType::F32).unwrap();
    let true_score_vecs: Vec<Vec<f32>> = q_tensors
        .iter()
        .map(|q| {
            q.to_dtype(DType::F32)
                .unwrap()
                .matmul(&k_f32.transpose(2, 3).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        })
        .collect();

    let compute_recalls = |pred_score_vecs: &[Vec<f32>]| -> (f32, f32, f32) {
        let mut r1 = 0.0f32;
        let mut r5 = 0.0f32;
        let mut r10 = 0.0f32;
        for (true_s, pred_s) in true_score_vecs.iter().zip(pred_score_vecs.iter()) {
            let true1 = top_k_indices(true_s, 1);
            let true5 = top_k_indices(true_s, 5);
            let true10 = top_k_indices(true_s, 10);
            let pred1 = top_k_indices(pred_s, 1);
            let pred5 = top_k_indices(pred_s, 5);
            let pred10 = top_k_indices(pred_s, 10);
            r1 += recall_at_k(&true1, &pred1);
            r5 += recall_at_k(&true5, &pred5);
            r10 += recall_at_k(&true10, &pred10);
        }
        let n = n_queries as f32;
        (r1 / n, r5 / n, r10 / n)
    };

    {
        let (r1, r5, r10) = compute_recalls(&true_score_vecs);
        println!("║ FullPrec (16) ║ {r1:>9.3}     ║ {r5:>9.3}     ║ {r10:>9.3}     ║  16.0  ║");
    }

    {
        let cfg = TurboQuantConfig::new_3p5bit(d, seed_base);
        let turbo_keys = turbo_quantize_tensor(&k_tensor, &cfg, 0).unwrap();
        let avg_bpd = if num_heads > 0 && !turbo_keys[0].is_empty() {
            turbo_keys[0][0].bits_per_dim()
        } else {
            0.0
        };

        let pred_vecs: Vec<Vec<f32>> = q_tensors
            .iter()
            .map(|q| {
                turbo_attention_scores(q, &turbo_keys, &cfg)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect();
        let (r1, r5, r10) = compute_recalls(&pred_vecs);
        println!(
            "║ TurboQ 3.5b   ║ {r1:>9.3}     ║ {r5:>9.3}     ║ {r10:>9.3}     ║ {avg_bpd:>4.1}   ║"
        );
    }

    println!("╚═══════════════╩═══════════════╩═══════════════╩═══════════════╩════════╝");
}

fn main() {
    println!("TurboQuant Runtime Evaluation");

    let device = if candle::utils::cuda_is_available() {
        candle::Device::new_cuda(0).unwrap_or(candle::Device::Cpu)
    } else {
        candle::Device::Cpu
    };

    run_compression_benchmark();
    run_accuracy_benchmark();
    run_recall_benchmark(&device);
}
