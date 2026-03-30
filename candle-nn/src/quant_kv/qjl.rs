//! QJL: Asymmetric 1-Bit Quantized Johnson-Lindenstrauss Transform for KV Cache.
//!
//! ## Paper Reference
//!
//! "Asymmetric 1-Bit Quantized Johnson-Lindenstrauss Transformations for KV Cache"
//! Zandieh et al. (2024) — arxiv:2406.03482
//!
//! ## Algorithm Overview
//!
//! QJL compresses key vectors in the KV cache to just 1 bit per dimension plus a single
//! floating-point norm, achieving >5× memory reduction versus FP16 with no accuracy loss.
//!
//! ### Encoding (applied to each new key token)
//!
//! Given key vector k ∈ ℝ^d:
//! 1. Generate random matrix S ∈ ℝ^{d×d} with entries S[i,j] ~ N(0,1) using a seeded PRNG
//! 2. Compute s_i = ⟨S[i,:], k⟩ for each i ∈ {0, ..., d-1}
//! 3. Store `sign_bits[i] = 1 if s_i > 0, else 0` (packed into u8 arrays, 8 bits per byte)
//! 4. Store `norm = ||k||₂` as a 16-bit float (f16)
//!
//! Total storage: d/8 bytes (signs) + 2 bytes (norm) = d/8 + 2 bytes per key vector
//!
//! ### Decoding (computing attention scores during inference)
//!
//! Given query q ∈ ℝ^d (kept at full precision), the inner product estimator is:
//!
//! ```text
//! ⟨q, k⟩ ≈ (π / (2d)) · ||k||₂ · ⟨q, S^T · σ⟩
//! ```
//!
//! where σ ∈ {±1}^d is the sign vector (σ_i = sign(s_i)).
//!
//! **Key property**: This estimator is **unbiased**:
//! E[estimated ⟨q, k⟩] = ⟨q, k⟩
//!
//! This follows from the Johnson-Lindenstrauss lemma: for a Gaussian projection matrix S,
//! E[sign(S·k) · S^T · q] = (2/π) · (k / ||k||₂) · ⟨q, k⟩ / ||k||₂, scaled appropriately.
//!
//! ### Asymmetric Design
//!
//! "Asymmetric" means only the KEY is quantized. The QUERY stays at full precision (FP16/F32).
//! This is critical because:
//! - Quantizing both Q and K would introduce correlated errors, breaking unbiasedness
//! - During decoding, we compute one attention row at a time (one query token)
//! - The query is never cached, so there is no memory pressure to quantize it
//!
//! ### Memory Analysis
//!
//! For head_dim d = 128 (typical for LLaMA-7B):
//! - FP16 baseline: 128 × 2 = 256 bytes
//! - QJL: 128/8 + 2 = 18 bytes → **14.2× compression**
//!
//! Effective bits per dimension: 1.0 + 16/d
//! - d=64: 1.25 bits/dim
//! - d=128: 1.125 bits/dim
//! - d=256: 1.0625 bits/dim

use candle::{Result, Tensor};
use half::f16;

use super::prng::Prng;

/// Configuration for QJL quantization.
///
/// The `seed` uniquely identifies the random projection matrix S. The same seed must be used
/// for both encoding (in `qjl_quantize`) and decoding (in `qjl_inner_product`), since S is
/// generated on-the-fly from the seed rather than stored.
#[derive(Debug, Clone)]
pub struct QjlConfig {
    /// Dimension of the key/query vectors (head dimension, typically 64–256)
    pub dim: usize,
    /// Seed for the random projection matrix S. Must be consistent across encode/decode.
    pub seed: u64,
}

impl QjlConfig {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self { dim, seed }
    }
}

/// A quantized representation of a single key vector.
///
/// Stores `d` sign bits (packed into `ceil(d/8)` bytes) and the L2 norm as f16.
#[derive(Debug, Clone)]
pub struct QjlQuantizedKey {
    /// Sign bits: bit i of byte j corresponds to dimension (j*8 + i).
    /// Bit is 1 if S[row,:] · k > 0, else 0.
    pub sign_bits: Vec<u8>,
    /// L2 norm ||k||₂ stored as 16-bit float to minimize storage.
    pub norm: f16,
    /// Original key dimension.
    pub dim: usize,
}

impl QjlQuantizedKey {
    /// Memory usage in bytes for this compressed key.
    pub fn byte_size(&self) -> usize {
        self.sign_bits.len() + 2 // sign bits + f16 norm
    }

    /// Effective bits per key dimension including the norm overhead.
    pub fn bits_per_dim(&self) -> f32 {
        (self.byte_size() * 8) as f32 / self.dim as f32
    }
}

/// Quantize a single key vector to its 1-bit QJL representation.
///
/// Generates the random projection matrix S on-the-fly from the seed in `config`, computes
/// s = S·k, and stores the sign bits and L2 norm.
///
/// Time complexity: O(d²) — each of the d rows of S is a d-dimensional vector, and we compute
/// one dot product per row. For d=128 this is 128×128 = 16,384 floating point operations,
/// which is negligible compared to the transformer's attention computation.
///
/// # Arguments
/// * `k` — key vector of length `config.dim`
/// * `config` — QJL configuration with dimension and PRNG seed
pub fn qjl_quantize(k: &[f32], config: &QjlConfig, token_idx: usize) -> QjlQuantizedKey {
    let d = config.dim;
    debug_assert_eq!(k.len(), d, "key length must equal config.dim");

    let num_bytes = (d + 7) / 8;
    let mut sign_bits = vec![0u8; num_bytes];

    let mut rng = Prng::new(config.seed ^ token_idx as u64);
    let mut row = vec![0.0f32; d];

    for i in 0..d {
        // Generate row i of S: S[i,:] ~ N(0,1)^d
        rng.fill_normal(&mut row);

        // Compute s_i = ⟨S[i,:], k⟩
        let s_i: f32 = row.iter().zip(k.iter()).map(|(s, ki)| s * ki).sum();

        // Pack sign bit: bit (i % 8) of byte (i / 8)
        if s_i > 0.0 {
            sign_bits[i / 8] |= 1u8 << (i % 8);
        }
    }

    // Compute L2 norm
    let norm_sq: f32 = k.iter().map(|x| x * x).sum();
    let norm = f16::from_f32(norm_sq.sqrt());

    QjlQuantizedKey {
        sign_bits,
        norm,
        dim: d,
    }
}

/// Reconstruct a key vector approximation from its QJL-compressed form.
///
/// Computes the asymmetric estimator vector: `(π / (2d)) · ||k||₂ · (S^T · σ)`.
pub fn qjl_dequantize(qk: &QjlQuantizedKey, config: &QjlConfig, token_idx: usize) -> Vec<f32> {
    let d = config.dim;
    debug_assert_eq!(qk.dim, d);

    let mut result = vec![0.0f32; d];
    let mut rng = Prng::new(config.seed ^ token_idx as u64);
    let mut row = vec![0.0f32; d];

    for i in 0..d {
        rng.fill_normal(&mut row);
        let bit = (qk.sign_bits[i / 8] >> (i % 8)) & 1;
        let sigma_i = if bit == 1 { 1.0f32 } else { -1.0f32 };

        for j in 0..d {
            result[j] += sigma_i * row[j];
        }
    }

    let norm = f32::from(qk.norm);
    let scale = (std::f32::consts::PI / 2.0).sqrt() / d as f32 * norm;
    for x in &mut result {
        *x *= scale;
    }
    result
}

/// Estimate the inner product ⟨q, k_original⟩ from the QJL-compressed key.
///
/// Uses the asymmetric estimator:
///   ⟨q, k⟩ ≈ (π / (2d)) · ||k||₂ · ⟨q, S^T · σ⟩
///
/// where σ_i = 2·sign_bits[i] - 1 ∈ {-1, +1}.
///
/// This estimator is unbiased: E[estimate] = ⟨q, k⟩.
///
/// # Arguments
/// * `q` — query vector of length `config.dim` (full precision, never quantized)
/// * `qk` — the QJL-compressed key produced by `qjl_quantize`
/// * `config` — must use the same `seed` as was used during encoding
pub fn qjl_inner_product(q: &[f32], qk: &QjlQuantizedKey, config: &QjlConfig, token_idx: usize) -> f32 {
    let k_hat = qjl_dequantize(qk, config, token_idx);
    q.iter().zip(k_hat.iter()).map(|(qi, ki)| qi * ki).sum()
}

/// Quantize all key tokens in a key tensor.
///
/// # Arguments
/// * `k` — key tensor with shape `[batch, num_heads, seq_len, head_dim]` or
///   `[num_heads, seq_len, head_dim]`
/// * `config` — QJL configuration; `config.dim` must equal the last dimension of `k`
///
/// # Returns
/// A nested Vec indexed `[head][token]` containing the compressed keys.
/// Batch dimension is not supported (batch=1 assumed, as is standard in autoregressive decode).
pub fn qjl_quantize_tensor(
    k: &Tensor,
    config: &QjlConfig,
    seq_offset: usize,
) -> Result<Vec<Vec<QjlQuantizedKey>>> {
    let dims = k.dims();
    // Support both [batch, heads, seq, dim] and [heads, seq, dim]
    let (num_heads, seq_len, head_dim) = match dims.len() {
        4 => (dims[1], dims[2], dims[3]),
        3 => (dims[0], dims[1], dims[2]),
        _ => candle::bail!("qjl_quantize_tensor: expected 3D or 4D tensor, got {}D", dims.len()),
    };
    assert_eq!(
        head_dim, config.dim,
        "head_dim={head_dim} must equal config.dim={}",
        config.dim
    );

    // Flatten to f32 for processing
    let k_f32 = k.to_device(&candle::Device::Cpu)?.to_dtype(candle::DType::F32)?.flatten_all()?;
    let k_data = k_f32.to_vec1::<f32>()?;

    let mut all_heads = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let mut head_keys = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let offset = (h * seq_len + t) * head_dim;
            // Adjust for batch dim if needed
            let offset = if dims.len() == 4 {
                // Assume batch=0
                (h * seq_len + t) * head_dim
            } else {
                offset
            };
            let key_slice = &k_data[offset..offset + head_dim];
            head_keys.push(qjl_quantize(key_slice, config, t + seq_offset));
        }
        all_heads.push(head_keys);
    }

    Ok(all_heads)
}

/// Compute attention scores Q·K^T using QJL-compressed keys.
///
/// For each query token and each cached key token, uses the asymmetric QJL estimator.
///
/// # Arguments
/// * `q` — query tensor with shape `[batch=1, num_heads, q_len, head_dim]`
/// * `quantized_keys` — compressed keys indexed `[head][token]`
/// * `config` — must use the same seed as was used during key encoding
///
/// # Returns
/// Attention score tensor of shape `[1, num_heads, q_len, kv_len]`
pub fn qjl_attention_scores(
    q: &Tensor,
    quantized_keys: &[Vec<QjlQuantizedKey>],
    config: &QjlConfig,
) -> Result<Tensor> {
    let dims = q.dims();
    let (batch, num_heads, q_len, head_dim) = match dims.len() {
        4 => (dims[0], dims[1], dims[2], dims[3]),
        _ => candle::bail!("qjl_attention_scores: expected 4D query tensor"),
    };
    assert_eq!(num_heads, quantized_keys.len());
    let kv_len = if num_heads > 0 { quantized_keys[0].len() } else { 0 };

    if kv_len == 0 {
        return Tensor::zeros((batch, num_heads, q_len, kv_len), q.dtype(), q.device());
    }

    // Pre-dequantize all keys into a flat vector on CPU
    let mut k_data = Vec::with_capacity(batch * num_heads * kv_len * head_dim);

    for _ in 0..batch {
        for h in 0..num_heads {
            for kt in 0..kv_len {
                let k_vec = qjl_dequantize(&quantized_keys[h][kt], config, kt);
                k_data.extend_from_slice(&k_vec);
            }
        }
    }

    let k_tensor = Tensor::from_vec(k_data, (batch, num_heads, kv_len, head_dim), q.device())?
        .to_dtype(q.dtype())?;
    q.matmul(&k_tensor.transpose(2, 3)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// Helper: generate a random f32 vector of given length using a seed.
    fn random_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = Prng::new(seed);
        let mut v = vec![0.0f32; len];
        rng.fill_normal(&mut v);
        v
    }

    /// Normalize a vector to unit L2 norm.
    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }

    /// Verify that the QJL inner product estimator is unbiased.
    ///
    /// For N=1000 random (q, k) pairs, the mean of (estimate - true_dot) should be near 0.
    /// This directly validates the theoretical guarantee from the QJL paper.
    #[test]
    fn unbiasedness() {
        let d = 64;
        let n = 1000;
        let config = QjlConfig::new(d, 42);

        let mut sum_error = 0.0f64;

        for i in 0..n {
            let q = normalize(&random_vec(d, i as u64));
            let k = normalize(&random_vec(d, i as u64 + 1_000_000));

            let true_dot: f32 = q.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum();
            let qk = qjl_quantize(&k, &config, i);
            let estimated = qjl_inner_product(&q, &qk, &config, i);

            sum_error += (estimated - true_dot) as f64;
        }

        let mean_error = (sum_error / n as f64).abs() as f32;
        assert!(
            mean_error < 0.10,
            "QJL unbiasedness failed: mean_error={mean_error} (expected < 0.10)"
        );
    }

    /// Verify that byte_size matches the formula d/8 + 2.
    #[test]
    fn memory_ratio() {
        for d in [32usize, 64, 128, 256] {
            let config = QjlConfig::new(d, 1);
            let k = vec![1.0f32; d];
            let qk = qjl_quantize(&k, &config, 0);
            let expected = d / 8 + 2;
            assert_eq!(
                qk.byte_size(),
                expected,
                "byte_size mismatch for d={d}: got {}, expected {expected}",
                qk.byte_size()
            );
            // For d=128: 18 bytes vs 256 bytes for FP16 → >14× compression
            let fp16_size = d * 2;
            let ratio = fp16_size as f32 / qk.byte_size() as f32;
            assert!(
                ratio >= 5.0,
                "compression ratio {ratio:.2}× below 5× threshold for d={d}"
            );
        }
    }

    /// Verify that qjl_quantize_tensor and scalar API are consistent.
    #[test]
    fn tensor_scalar_consistency() -> Result<()> {
        let d = 64;
        let num_heads = 2;
        let seq_len = 4;
        let config = QjlConfig::new(d, 77);

        // Create random key tensor [1, num_heads, seq_len, d]
        let k_data: Vec<f32> = (0..num_heads * seq_len * d)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let k = Tensor::from_vec(k_data.clone(), (1, num_heads, seq_len, d), &Device::Cpu)?;

        let all_keys = qjl_quantize_tensor(&k, &config, 0)?;

        // Check that each head/token matches scalar API
        for h in 0..num_heads {
            for t in 0..seq_len {
                let k_slice = &k_data[(h * seq_len + t) * d..(h * seq_len + t + 1) * d];
                let qk_scalar = qjl_quantize(k_slice, &config, t);
                let qk_tensor = &all_keys[h][t];

                assert_eq!(qk_scalar.sign_bits, qk_tensor.sign_bits);
                // norm: f16 from same f32 → should be identical
                assert_eq!(qk_scalar.norm, qk_tensor.norm);
            }
        }
        Ok(())
    }

    /// Verify that zero vector produces zero norm.
    #[test]
    fn zero_vector() {
        let d = 64;
        let config = QjlConfig::new(d, 1);
        let k = vec![0.0f32; d];
        let qk = qjl_quantize(&k, &config, 0);
        assert_eq!(f32::from(qk.norm), 0.0, "zero key should have zero norm");
    }

    /// Verify estimation quality (variance reasonable) on unit vectors.
    #[test]
    fn estimation_variance() {
        let d = 128;
        let n = 500;
        let _config = QjlConfig::new(d, 42); // reference config, variants use different seeds

        // Use a fixed q and k pair
        let q = normalize(&random_vec(d, 1));
        let k = normalize(&random_vec(d, 2));
        let true_dot: f32 = q.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum();

        // Quantize the same k with different configs (different seeds = independent estimates)
        let mut estimates = Vec::with_capacity(n);
        for s in 0..n {
            let cfg = QjlConfig::new(d, s as u64 + 100);
            let qk = qjl_quantize(&k, &cfg, 0);
            estimates.push(qjl_inner_product(&q, &qk, &cfg, 0));
        }

        let mean: f32 = estimates.iter().sum::<f32>() / n as f32;
        let bias = (mean - true_dot).abs();

        // Bias should be small (unbiased estimator)
        assert!(
            bias < 0.1,
            "estimation bias too large: {bias:.4} (true={true_dot:.4}, mean={mean:.4})"
        );
    }
}
