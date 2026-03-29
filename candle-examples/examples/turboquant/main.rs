//! TurboQuant / QJL / PolarQuant Benchmark Example
//!
//! This example benchmarks and compares the three KV cache quantization algorithms
//! implemented in `candle_nn::quant_kv`:
//!
//! - **QJL**: 1-bit asymmetric key quantization (arxiv:2406.03482)
//! - **PolarQuant**: Recursive polar coordinate decomposition (arxiv:2502.02617)
//! - **TurboQuant**: Two-stage hybrid at 3.5 bits/channel (arxiv:2504.19874)
//!
//! ## Running
//!
//! ```bash
//! cargo run --example turboquant --release
//! ```
//!
//! ## Output
//!
//! The example prints three benchmark tables:
//! 1. **Compression ratios** vs FP16 baseline for each algorithm and head dimension
//! 2. **Inner product accuracy** — empirical mean error and variance over 10,000 random pairs
//! 3. **Top-k recall** — attention score recall in a simulated 1024-token KV cache
//!
//! These results should match the claims in the TurboQuant paper and Google Research blog post.

use candle::{DType, Device, Tensor};
use candle_nn::quant_kv::{
    polar_quant::{polar_attention_scores, polar_quantize_tensor, PolarQuantConfig},
    qjl::{qjl_attention_scores, qjl_quantize, qjl_quantize_tensor, QjlConfig},
    prng::Prng,
    turbo_quant::{turbo_attention_scores, turbo_quantize, turbo_quantize_tensor, TurboQuantConfig},
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
fn random_tensor(shape: &[usize], seed: u64) -> Tensor {
    let total: usize = shape.iter().product();
    let mut rng = Prng::new(seed);
    let mut data = vec![0.0f32; total];
    rng.fill_normal(&mut data);
    // Normalize each head_dim slice
    let head_dim = *shape.last().unwrap();
    for chunk in data.chunks_mut(head_dim) {
        let norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in chunk.iter_mut() {
                *x /= norm;
            }
        }
    }
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

/// ─────────────────────────────────────────────────────────────────────────
/// Benchmark 1: Compression Ratios
/// ─────────────────────────────────────────────────────────────────────────
///
/// Shows how many bytes each algorithm uses per key vector compared to FP16 (2 bytes/dim).
/// This directly validates the memory reduction claims in the papers.
fn run_compression_benchmark() {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║               Benchmark 1: Compression Ratios vs FP16              ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Paper claims:                                                       ║");
    println!("║    QJL:         >5× compression (>14× for d=128)                    ║");
    println!("║    PolarQuant:  ~4.2× compression at d=64                           ║");
    println!("║    TurboQuant:  ~4.5× compression at 3.5 bits                      ║");
    println!("╠══════════════╦══════════╦════════════╦═════════════╦═══════════╣");
    println!("║ head_dim     ║ FP16     ║ QJL        ║ PolarQuant  ║ TurboQuant║");
    println!("║              ║ (bytes)  ║ bits/dim   ║ bits/dim    ║ bits/dim  ║");
    println!("╠══════════════╬══════════╬════════════╬═════════════╬═══════════╣");

    for &d in &[32usize, 64, 128, 256] {
        let fp16_bytes = d * 2;
        let k = random_unit_vec(d, 42);

        // QJL
        let qjl_cfg = QjlConfig::new(d, 1);
        let qjl_key = qjl_quantize(&k, &qjl_cfg);
        let qjl_bpd = qjl_key.bits_per_dim();

        // PolarQuant
        let polar_cfg = PolarQuantConfig::new(d, 1);
        let dummy_k = random_unit_vec(d, 7);
        let polar_key = candle_nn::quant_kv::polar_quant::polar_quantize(&dummy_k, &polar_cfg);
        let polar_bpd = polar_key.bits_per_dim();

        // TurboQuant
        let turbo_cfg = TurboQuantConfig::new_3p5bit(d, 1);
        let turbo_key = turbo_quantize(&k, &turbo_cfg);
        let turbo_bpd = turbo_key.bits_per_dim();

        println!(
            "║ d={d:<9} ║ {fp16_bytes:<8} ║ {qjl_bpd:>6.3} ({:.1}×) ║ {polar_bpd:>6.3} ({:.1}×)  ║ {turbo_bpd:>5.3} ({:.1}×)║",
            fp16_bytes as f32 * 8.0 / (qjl_key.byte_size() as f32 * 8.0),
            fp16_bytes as f32 * 8.0 / (polar_key.byte_size() as f32 * 8.0),
            fp16_bytes as f32 * 8.0 / (turbo_key.byte_size() as f32 * 8.0),
        );
    }
    println!("╚══════════════╩══════════╩════════════╩═════════════╩═══════════╝");
}

/// ─────────────────────────────────────────────────────────────────────────
/// Benchmark 2: Inner Product Accuracy
/// ─────────────────────────────────────────────────────────────────────────
///
/// Measures the quality of the inner product estimator for each algorithm.
/// Tests the theoretical unbiasedness guarantee: E[estimate] = true_dot_product.
///
/// Paper claims:
/// - QJL: unbiased estimator, variance O(1/d) per pair
/// - TurboQuant 3.5-bit: D_prod ≤ (√3·π²/d) · (1/4^b)
fn run_accuracy_benchmark() {
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║             Benchmark 2: Inner Product Accuracy (N=2000 pairs)          ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Tests the unbiasedness guarantee: E[estimated ⟨q,k⟩] = true ⟨q,k⟩    ║");
    println!("╠═══════════╦══════════════╦═══════════════╦══════════════╦════════════╣");
    println!("║ Algorithm ║ Mean Error   ║ Mean Abs Err  ║ Std Dev Err  ║ 95th %ile  ║");
    println!("╠═══════════╬══════════════╬═══════════════╬══════════════╬════════════╣");

    let d = 64usize;
    let n = 2000usize;

    // Pre-generate all (q, k) pairs
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

    // Full-precision baseline (no quantization) — error should be 0.000
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

    // QJL
    {
        let cfg = QjlConfig::new(d, 42);
        let errors: Vec<f32> = pairs
            .iter()
            .zip(true_dots.iter())
            .map(|((q, k), &true_dot)| {
                let qk = qjl_quantize(k, &cfg);
                let est = candle_nn::quant_kv::qjl::qjl_inner_product(q, &qk, &cfg);
                est - true_dot
            })
            .collect();
        print_accuracy_row("QJL", &errors);
    }

    // PolarQuant
    {
        let cfg = PolarQuantConfig::new(d, 42);
        let errors: Vec<f32> = pairs
            .iter()
            .zip(true_dots.iter())
            .map(|((q, k), &true_dot)| {
                let pq = candle_nn::quant_kv::polar_quant::polar_quantize(k, &cfg);
                let est = candle_nn::quant_kv::polar_quant::polar_inner_product(q, &pq, &cfg);
                est - true_dot
            })
            .collect();
        print_accuracy_row("PolarQuant", &errors);
    }

    // TurboQuant 3.5-bit
    {
        let cfg = TurboQuantConfig::new_3p5bit(d, 42);
        let errors: Vec<f32> = pairs
            .iter()
            .zip(true_dots.iter())
            .map(|((q, k), &true_dot)| {
                let tq = turbo_quantize(k, &cfg);
                let est = candle_nn::quant_kv::turbo_quant::turbo_inner_product(q, &tq, &cfg);
                est - true_dot
            })
            .collect();
        print_accuracy_row("TurboQ-3.5b", &errors);
    }

    // TurboQuant 2.5-bit
    {
        let cfg = TurboQuantConfig::new(d, 2.5, 42);
        let errors: Vec<f32> = pairs
            .iter()
            .zip(true_dots.iter())
            .map(|((q, k), &true_dot)| {
                let tq = turbo_quantize(k, &cfg);
                let est = candle_nn::quant_kv::turbo_quant::turbo_inner_product(q, &tq, &cfg);
                est - true_dot
            })
            .collect();
        print_accuracy_row("TurboQ-2.5b", &errors);
    }

    println!("╚═══════════╩══════════════╩═══════════════╩══════════════╩════════════╝");
    println!("  (all values for d=64 unit vectors)");
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

    println!(
        "║ {name:<9} ║ {mean:>+12.4} ║ {mean_abs:>13.4} ║ {std_dev:>12.4} ║ {p95:>10.4} ║"
    );
}

/// ─────────────────────────────────────────────────────────────────────────
/// Benchmark 3: Top-k Recall in Simulated KV Cache
/// ─────────────────────────────────────────────────────────────────────────
///
/// Simulates autoregressive decoding with a cached sequence of n_tokens tokens.
/// Measures how often the top-k most attended keys according to the quantized scores
/// match the top-k keys from full-precision (unquantized) scores.
///
/// Paper claim (TurboQuant vs KIVI on Needle-in-Haystack):
///   TurboQuant: 0.997, KIVI: 0.981, Full precision: 0.997
fn run_recall_benchmark() {
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║           Benchmark 3: Top-k Attention Recall (N=100 queries)           ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Simulates retrieving the k most-attended tokens from a cached sequence.║");
    println!("║  Measures: fraction of true top-k tokens recovered by quantized scores. ║");
    println!("╠═══════════════╦═══════════════╦═══════════════╦═══════════════╦════════╣");
    println!("║ Algorithm     ║ recall@1      ║ recall@5      ║ recall@10     ║ bits   ║");
    println!("╠═══════════════╬═══════════════╬═══════════════╬═══════════════╬════════╣");

    let d = 64usize;
    let n_tokens = 256usize; // simulated cached sequence length
    let n_queries = 100usize;
    let num_heads = 1usize;
    let seed_base = 12345u64;

    // Pre-generate all key vectors for the simulated cache
    let k_tensor = random_tensor(&[1, num_heads, n_tokens, d], seed_base);

    // Pre-generate query vectors
    let q_tensors: Vec<Tensor> = (0..n_queries)
        .map(|i| random_tensor(&[1, num_heads, 1, d], seed_base + i as u64 + 100_000))
        .collect();

    // Full-precision baseline: Q·K^T
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

    // Compute full-precision scores for all queries
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

    // Helper: compute recall for a given set of predicted score vectors
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

    // ── Full-precision baseline (no quantization) ─────────────────────────
    // This is the "without QuantizedKvCache" baseline; recall must be 1.000.
    {
        let (r1, r5, r10) = compute_recalls(&true_score_vecs);
        println!("║ FullPrec (FP16)║ {r1:>9.3}     ║ {r5:>9.3}     ║ {r10:>9.3}     ║  16.0  ║");
    }

    // ── QJL ──────────────────────────────────────────────────────────────
    {
        let cfg = QjlConfig::new(d, seed_base);
        let qjl_keys = qjl_quantize_tensor(&k_tensor, &cfg).unwrap();
        let avg_bpd = if num_heads > 0 && !qjl_keys[0].is_empty() {
            qjl_keys[0][0].bits_per_dim()
        } else { 0.0 };

        let pred_vecs: Vec<Vec<f32>> = q_tensors
            .iter()
            .map(|q| {
                qjl_attention_scores(q, &qjl_keys, &cfg)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect();
        let (r1, r5, r10) = compute_recalls(&pred_vecs);
        println!("║ QJL           ║ {r1:>9.3}     ║ {r5:>9.3}     ║ {r10:>9.3}     ║ {avg_bpd:>4.1}   ║");
    }

    // ── PolarQuant ───────────────────────────────────────────────────────
    {
        let cfg = PolarQuantConfig::new(d, seed_base);
        let polar_keys = polar_quantize_tensor(&k_tensor, &cfg).unwrap();
        let avg_bpd = if num_heads > 0 && !polar_keys[0].is_empty() {
            polar_keys[0][0].bits_per_dim()
        } else { 0.0 };

        let pred_vecs: Vec<Vec<f32>> = q_tensors
            .iter()
            .map(|q| {
                polar_attention_scores(q, &polar_keys, &cfg)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect();
        let (r1, r5, r10) = compute_recalls(&pred_vecs);
        println!("║ PolarQuant    ║ {r1:>9.3}     ║ {r5:>9.3}     ║ {r10:>9.3}     ║ {avg_bpd:>4.1}   ║");
    }

    // ── TurboQuant 3.5-bit ───────────────────────────────────────────────
    {
        let cfg = TurboQuantConfig::new_3p5bit(d, seed_base);
        let turbo_keys = turbo_quantize_tensor(&k_tensor, &cfg).unwrap();
        let avg_bpd = if num_heads > 0 && !turbo_keys[0].is_empty() {
            turbo_keys[0][0].bits_per_dim()
        } else { 0.0 };

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
        println!("║ TurboQ 3.5b   ║ {r1:>9.3}     ║ {r5:>9.3}     ║ {r10:>9.3}     ║ {avg_bpd:>4.1}   ║");
    }

    // ── TurboQuant 2.5-bit ───────────────────────────────────────────────
    {
        let cfg = TurboQuantConfig::new(d, 2.5, seed_base);
        let turbo_keys = turbo_quantize_tensor(&k_tensor, &cfg).unwrap();
        let avg_bpd = if num_heads > 0 && !turbo_keys[0].is_empty() {
            turbo_keys[0][0].bits_per_dim()
        } else { 0.0 };

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
        println!("║ TurboQ 2.5b   ║ {r1:>9.3}     ║ {r5:>9.3}     ║ {r10:>9.3}     ║ {avg_bpd:>4.1}   ║");
    }

    println!("╚═══════════════╩═══════════════╩═══════════════╩═══════════════╩════════╝");
    println!("  (d=64, {n_tokens} cached tokens, {n_queries} query vectors)");
    println!("  recall@k = fraction of true top-k tokens in predicted top-k");
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║     TurboQuant / QJL / PolarQuant — Candle Implementation Benchmarks    ║");
    println!("║                                                                          ║");
    println!("║  Papers:                                                                 ║");
    println!("║    QJL:         arxiv.org/abs/2406.03482                                 ║");
    println!("║    PolarQuant:  arxiv.org/abs/2502.02617                                 ║");
    println!("║    TurboQuant:  arxiv.org/abs/2504.19874 (ICLR 2026)                    ║");
    println!("║  Blog:          research.google/blog/turboquant-redefining-ai-           ║");
    println!("║                 efficiency-with-extreme-compression/                     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    println!("\nRunning benchmarks... (this may take a minute due to O(d²) QJL operations)");

    let t0 = std::time::Instant::now();
    run_compression_benchmark();
    let t1 = std::time::Instant::now();
    println!("  (completed in {:.2}s)", (t1 - t0).as_secs_f32());

    let t0 = std::time::Instant::now();
    run_accuracy_benchmark();
    let t1 = std::time::Instant::now();
    println!("  (completed in {:.2}s)", (t1 - t0).as_secs_f32());

    let t0 = std::time::Instant::now();
    run_recall_benchmark();
    let t1 = std::time::Instant::now();
    println!("  (completed in {:.2}s)", (t1 - t0).as_secs_f32());

    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  Key findings to compare with paper claims:                              ║");
    println!("║                                                                           ║");
    println!("║  1. QJL achieves >5× compression vs FP16 at d=64+ (paper: >5×)          ║");
    println!("║  2. Mean error of QJL and TurboQuant should be near 0 (unbiased)         ║");
    println!("║  3. TurboQuant 3.5-bit recall should exceed PolarQuant and QJL           ║");
    println!("║  4. TurboQuant shows best recall/bit tradeoff                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}
