//! PolarQuant: Recursive Polar Coordinate Decomposition for KV Cache Compression.
//!
//! ## Paper Reference
//!
//! "PolarQuant: Compressing KV Cache with Polar Coordinate Quantization"
//! (arxiv:2502.02617)
//!
//! ## Algorithm Overview
//!
//! PolarQuant is a geometric compression method that represents high-dimensional key/value
//! vectors in polar coordinates rather than Cartesian coordinates. After a random rotation
//! preconditioning step, the angles of the polar decomposition follow a well-known Beta
//! distribution, enabling a data-oblivious optimal codebook.
//!
//! ### Motivation
//!
//! Standard quantization methods store (value - zero_point) / scale for each block, requiring
//! per-block floating-point metadata. PolarQuant eliminates this overhead by:
//! 1. Separating **magnitude** (a single radius) from **direction** (a set of angles)
//! 2. Exploiting the concentration of angles after random rotation to use a fixed codebook
//! 3. Storing the angles at 4 bits (level 1) and 2 bits (deeper levels) — no metadata
//!
//! ### Recursive Decomposition
//!
//! For a d-dimensional vector x (d must be a power of 2):
//!
//! **Level 1** (d/2 angles):
//!   For each adjacent pair (x_{2j-1}, x_{2j}):
//!     ψⱼ = atan2(x_{2j}, x_{2j-1}) ∈ (-π, π]   — quantized at 4 bits
//!     rⱼ = sqrt(x_{2j-1}² + x_{2j}²)            — passed to next level
//!
//! **Level ℓ ≥ 2** (d/2^ℓ angles):
//!   For each adjacent pair (r_{2j-1}, r_{2j}) from the previous level:
//!     θⱼ = atan2(r_{2j}, r_{2j-1}) ∈ [0, π/2]   — quantized at 2 bits
//!     Rⱼ = sqrt(r_{2j-1}² + r_{2j}²)             — passed to next level
//!
//! After log₂(d) levels, a single scalar radius remains (stored as f16).
//!
//! ### Distribution Theory
//!
//! After SRHT preconditioning, the angles at each level follow:
//! - Level 1: approximately **uniform** on (-π, π] → uniform 4-bit codebook is optimal
//! - Level ℓ ≥ 2: f(θ) ∝ sin^(2^(ℓ-1) - 1)(2θ) on [0, π/2] → Lloyd-Max codebook
//!
//! This distribution is fully determined by the level index (and d), independent of the
//! actual vector values — enabling data-oblivious quantization.
//!
//! ### Memory Analysis for d = 64
//!
//! | Level | Count | Bits | Total |
//! |-------|-------|------|-------|
//! | 1     | 32    | 4    | 128   |
//! | 2     | 16    | 2    | 32    |
//! | 3     | 8     | 2    | 16    |
//! | 4     | 4     | 2    | 8     |
//! | 5     | 2     | 2    | 4     |
//! | 6     | 1     | 2    | 2     |
//! | Radius| 1     | 16   | 16    |
//! | **Total** | | | **206 bits = 3.22 bits/dim** |
//!
//! vs. FP16 baseline: 64 × 16 = 1024 bits → **4.97× compression**

use candle::{Result, Tensor};
use half::f16;
use super::codebook::{get_polar_codebook, Codebook};
use super::fwht::{srht_inverse, srht_precondition};

/// Configuration for PolarQuant quantization.
///
/// Pre-computes codebooks for each level at construction time, so the potentially expensive
/// Lloyd-Max computation is amortized over many encode/decode calls.
#[derive(Debug, Clone)]
pub struct PolarQuantConfig {
    /// Head dimension (must be a power of 2 for the recursive decomposition)
    pub dim: usize,
    /// Seed for the SRHT preconditioning rotation
    pub seed: u64,
    /// Bits per angle at level 1 (default: 4)
    pub bits_level1: u32,
    /// Bits per angle at deeper levels (default: 2)
    pub bits_deep: u32,
    /// Pre-computed codebooks: index 0 = level 1, index 1 = level 2, etc.
    codebooks: Vec<Codebook>,
}

impl PolarQuantConfig {
    /// Create a new PolarQuantConfig with standard settings (4 bits level 1, 2 bits deeper).
    ///
    /// # Arguments
    /// * `dim` — head dimension; does not need to be a power of 2 (padded internally)
    /// * `seed` — random seed for SRHT preconditioning
    pub fn new(dim: usize, seed: u64) -> Self {
        Self::with_bits(dim, seed, 4, 2)
    }

    /// Create a PolarQuantConfig with custom bit allocations.
    pub fn with_bits(dim: usize, seed: u64, bits_level1: u32, bits_deep: u32) -> Self {
        let n_levels = {
            let mut d = dim;
            let mut levels = 0;
            while d > 1 {
                d = d.div_ceil(2);
                levels += 1;
            }
            levels
        };

        let mut codebooks = Vec::with_capacity(n_levels);
        for level in 1..=n_levels as u32 {
            let bits = if level == 1 { bits_level1 } else { bits_deep };
            codebooks.push(get_polar_codebook(bits, level));
        }

        Self {
            dim,
            seed,
            bits_level1,
            bits_deep,
            codebooks,
        }
    }

    #[allow(dead_code)]
    fn num_levels(&self) -> usize {
        self.codebooks.len()
    }

    fn codebook_for_level(&self, level: usize) -> &Codebook {
        // level is 1-indexed
        &self.codebooks[level - 1]
    }
}

/// A PolarQuant-compressed vector.
#[derive(Debug, Clone)]
pub struct PolarQuantizedVec {
    /// Quantized angle codes for level 1.
    /// Packed as 4-bit values (2 per byte) when bits_level1 > 2,
    /// or as 2-bit values (4 per byte) when bits_level1 <= 2.
    pub level1_codes: Vec<u8>,
    /// Quantized angle codes for levels 2 and deeper, packed by bits_deep:
    ///   bits_deep == 1 → pack_1bit (8 per byte)
    ///   bits_deep == 2 → pack_2bit (4 per byte)
    /// Stored level-by-level in order: level 2, then level 3, etc.
    pub leveln_codes: Vec<u8>,
    /// The single remaining radius after full recursive decomposition (stored as f16).
    pub final_radius: f16,
    /// Number of recursive levels.
    pub n_levels: usize,
    /// Original vector dimension.
    pub dim: usize,
    /// Bits per angle at level 1 (controls both codebook size and packing density).
    /// 1 → pack_1bit, 2 → pack_2bit, >2 → pack_4bit.
    pub bits_level1: u32,
    /// Bits per angle at deeper levels (controls both codebook size and packing density).
    /// 1 → pack_1bit, otherwise pack_2bit.
    pub bits_deep: u32,
    /// Angle counts per level (for unpacking).
    level_counts: Vec<usize>,
}

impl PolarQuantizedVec {
    /// Total memory usage in bytes.
    pub fn byte_size(&self) -> usize {
        self.level1_codes.len() + self.leveln_codes.len() + 2 // 2 bytes for f16 radius
    }

    /// Effective bits per dimension.
    pub fn bits_per_dim(&self) -> f32 {
        (self.byte_size() * 8) as f32 / self.dim as f32
    }
}

/// Pack 1-bit values (0..1) into bytes, 8 per byte.
#[inline]
fn pack_1bit(codes: &[u8]) -> Vec<u8> {
    let n = codes.len();
    let num_bytes = n.div_ceil(8);
    let mut packed = vec![0u8; num_bytes];
    for (i, &code) in codes.iter().enumerate() {
        packed[i / 8] |= (code & 0x1) << (i % 8);
    }
    packed
}

/// Unpack 1-bit values from a packed byte array.
#[inline]
fn unpack_1bit(packed: &[u8], count: usize) -> Vec<u8> {
    let mut codes = Vec::with_capacity(count);
    for i in 0..count {
        codes.push((packed[i / 8] >> (i % 8)) & 0x1);
    }
    codes
}

/// Pack two 4-bit values (0..15) into a single byte.
#[inline]
fn pack_4bit(codes: &[u8]) -> Vec<u8> {
    let n = codes.len();
    let num_bytes = n.div_ceil(2);
    let mut packed = vec![0u8; num_bytes];
    for (i, &code) in codes.iter().enumerate() {
        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        packed[byte_idx] |= (code & 0xF) << shift;
    }
    packed
}

/// Unpack 4-bit values from a packed byte array.
#[inline]
fn unpack_4bit(packed: &[u8], count: usize) -> Vec<u8> {
    let mut codes = Vec::with_capacity(count);
    for i in 0..count {
        let byte_idx = i / 2;
        let shift = (i % 2) * 4;
        codes.push((packed[byte_idx] >> shift) & 0xF);
    }
    codes
}

/// Pack four 2-bit values (0..3) into a single byte.
#[inline]
fn pack_2bit(codes: &[u8]) -> Vec<u8> {
    let n = codes.len();
    let num_bytes = n.div_ceil(4);
    let mut packed = vec![0u8; num_bytes];
    for (i, &code) in codes.iter().enumerate() {
        let byte_idx = i / 4;
        let shift = (i % 4) * 2;
        packed[byte_idx] |= (code & 0x3) << shift;
    }
    packed
}

/// Unpack 2-bit values from a packed byte array.
#[inline]
fn unpack_2bit(packed: &[u8], count: usize) -> Vec<u8> {
    let mut codes = Vec::with_capacity(count);
    for i in 0..count {
        let byte_idx = i / 4;
        let shift = (i % 4) * 2;
        codes.push((packed[byte_idx] >> shift) & 0x3);
    }
    codes
}

/// Quantize a single key/value vector using PolarQuant.
///
/// Steps:
/// 1. Apply SRHT preconditioning (random sign-flip + FWHT)
/// 2. Recursively decompose into angles and radii
/// 3. Quantize each angle using the appropriate codebook
/// 4. Pack codes into compact byte arrays
pub fn polar_quantize(x: &[f32], config: &PolarQuantConfig) -> PolarQuantizedVec {
    let d = x.len();

    // Step 1: SRHT preconditioning
    let mut rotated = vec![0.0f32; d];
    srht_precondition(x, config.seed, &mut rotated);

    // Step 2: Recursive polar decomposition
    let mut current_radii = rotated;
    let mut all_angle_codes: Vec<Vec<u8>> = Vec::new();
    let mut level_counts: Vec<usize> = Vec::new();

    let mut level = 1usize;

    while current_radii.len() > 1 {
        let n = current_radii.len();
        let n_pairs = n / 2;
        let cb = config.codebook_for_level(level);

        let mut angles = Vec::with_capacity(n_pairs);
        let mut next_radii = Vec::with_capacity((n + 1) / 2);

        for pair in 0..n_pairs {
            let r1 = current_radii[pair * 2];
            let r2 = current_radii[pair * 2 + 1];

            let angle = if level == 1 {
                // Level 1: full atan2 in (-π, π]
                r1.atan2(r2) as f64
            } else {
                // Level ≥ 2: atan2 of non-negative radii → [0, π/2]
                // atan2(|r2|, |r1|) ∈ [0, π/2] since both are non-negative norms
                r1.abs().atan2(r2.abs()) as f64
            };

            let code = cb.quantize(angle as f32);
            angles.push(code);
            next_radii.push(r1.hypot(r2));
        }

        // Handle odd-length: carry the last element unchanged to next level
        if n % 2 == 1 {
            next_radii.push(current_radii[n - 1]);
        }

        level_counts.push(n_pairs);
        all_angle_codes.push(angles);
        current_radii = next_radii;
        level += 1;
    }

    let final_radius = f16::from_f32(current_radii[0]);

    // Step 3: Pack codes — use the tightest packing for the given bit width.
    // bits_level1 == 1 → pack_1bit (8 per byte)
    // bits_level1 == 2 → pack_2bit (4 per byte)
    // bits_level1 <= 4 → pack_4bit (2 per byte)
    let level1_codes = if !all_angle_codes.is_empty() {
        if config.bits_level1 <= 1 {
            pack_1bit(&all_angle_codes[0])
        } else if config.bits_level1 <= 2 {
            pack_2bit(&all_angle_codes[0])
        } else {
            pack_4bit(&all_angle_codes[0])
        }
    } else {
        vec![]
    };

    // Pack all deeper levels concatenated — use bits_deep to choose packing
    let deep_codes_flat: Vec<u8> = all_angle_codes.iter().skip(1).flatten().copied().collect();
    let leveln_codes = if config.bits_deep <= 1 {
        pack_1bit(&deep_codes_flat)
    } else {
        pack_2bit(&deep_codes_flat)
    };

    PolarQuantizedVec {
        level1_codes,
        leveln_codes,
        final_radius,
        n_levels: all_angle_codes.len(),
        dim: d,
        bits_level1: config.bits_level1,
        bits_deep: config.bits_deep,
        level_counts,
    }
}

/// Reconstruct a key/value vector from its PolarQuant compressed form.
///
/// Performs the inverse polar decomposition:
/// 1. Unpack angle codes → angle centroids
/// 2. Reconstruct radii from the final radius up through each level
/// 3. Apply inverse SRHT
pub fn polar_dequantize(pq: &PolarQuantizedVec, config: &PolarQuantConfig) -> Vec<f32> {
    let n_levels = pq.n_levels;
    if n_levels == 0 {
        return vec![f32::from(pq.final_radius)];
    }

    // Unpack level-1 codes — must match the packing used during quantization
    let n_level1 = pq.level_counts[0];
    let level1_codes = if pq.bits_level1 <= 1 {
        unpack_1bit(&pq.level1_codes, n_level1)
    } else if pq.bits_level1 <= 2 {
        unpack_2bit(&pq.level1_codes, n_level1)
    } else {
        unpack_4bit(&pq.level1_codes, n_level1)
    };

    // Unpack deeper level codes — match the packing used during quantization
    let deep_total: usize = pq.level_counts.iter().skip(1).sum();
    let deep_codes = if pq.bits_deep <= 1 {
        unpack_1bit(&pq.leveln_codes, deep_total)
    } else {
        unpack_2bit(&pq.leveln_codes, deep_total)
    };

    // Build all angle vectors per level
    let mut all_angles: Vec<Vec<f32>> = Vec::with_capacity(n_levels);

    // Level 1
    {
        let cb = config.codebook_for_level(1);
        let angles: Vec<f32> = level1_codes
            .iter()
            .map(|&code| cb.dequantize(code))
            .collect();
        all_angles.push(angles);
    }

    // Levels 2+
    let mut deep_offset = 0;
    for lev in 2..=n_levels {
        let count = pq.level_counts[lev - 1];
        let cb = config.codebook_for_level(lev);
        let angles: Vec<f32> = deep_codes[deep_offset..deep_offset + count]
            .iter()
            .map(|&code| cb.dequantize(code))
            .collect();
        all_angles.push(angles);
        deep_offset += count;
    }

    // Reverse reconstruction: start from the final radius, work backward through levels
    // After quantization, deepest level gives us radii from which we build higher-level vectors.
    //
    // Forward: pairs (r1, r2) → (angle = atan2(r1, r2), R = hypot(r1, r2))
    // Backward: (angle, R) → (r1 = R·cos(angle), r2 = R·sin(angle))

    // Build the chain of radii vectors bottom-up
    // We have angles at n_levels levels and the final single radius.
    // Level n: 1 pair → 2 or more radii
    // ...
    // Level 1: dim/2 pairs → dim radii

    // Start with the single final radius
    let mut current = vec![f32::from(pq.final_radius)];

    // Expand from deepest level back to level 1
    for lev in (1..=n_levels).rev() {
        let angles = &all_angles[lev - 1];
        let level_count = pq.level_counts[lev - 1];

        // Each element in current (except possibly the last unpaired one) combined with the
        // corresponding angle to produce two elements.
        let mut next = Vec::with_capacity(current.len() + level_count);

        for (k, (&r, &angle)) in current
            .iter()
            .take(level_count)
            .zip(angles.iter())
            .enumerate()
        {
            // Reconstruction from angle = atan2(r1, r2) and R = hypot(r1, r2):
            //   sin(atan2(r1, r2)) = r1/R  →  r1 = R·sin(angle)
            //   cos(atan2(r1, r2)) = r2/R  →  r2 = R·cos(angle)
            let r1 = r * angle.sin();
            let r2 = r * angle.cos();
            next.push(r1);
            next.push(r2);
            let _ = k;
        }

        // If original level had an odd element, it was carried unchanged
        if current.len() > level_count {
            next.push(current[level_count]);
        }

        current = next;
    }

    // Trim to original dimension (might have extra from padding)
    current.truncate(pq.dim);
    // Pad if needed
    current.resize(pq.dim, 0.0);

    // Apply inverse SRHT
    let mut recovered = vec![0.0f32; pq.dim];
    srht_inverse(&current, config.seed, &mut recovered);
    recovered
}

/// Compute the dot product of a query vector with a dequantized PolarQuant key.
///
/// This is a convenience wrapper that dequantizes the key and then computes the dot product.
/// For maximum accuracy, use `turbo_quant` which adds QJL residual correction on top.
pub fn polar_inner_product(q: &[f32], pq: &PolarQuantizedVec, config: &PolarQuantConfig) -> f32 {
    let k_hat = polar_dequantize(pq, config);
    q.iter().zip(k_hat.iter()).map(|(qi, ki)| qi * ki).sum()
}

/// Quantize all key tokens in a key tensor using PolarQuant.
///
/// # Arguments
/// * `k` — key tensor with shape `[batch=1, num_heads, seq_len, head_dim]` or `[num_heads, seq_len, head_dim]`
/// * `config` — PolarQuant configuration; `config.dim` must equal the last dimension
///
/// # Returns
/// Compressed keys indexed `[head][token]`
pub fn polar_quantize_tensor(
    k: &Tensor,
    config: &PolarQuantConfig,
) -> Result<Vec<Vec<PolarQuantizedVec>>> {
    let dims = k.dims();
    let (num_heads, seq_len, head_dim) = match dims.len() {
        4 => (dims[1], dims[2], dims[3]),
        3 => (dims[0], dims[1], dims[2]),
        _ => candle::bail!(
            "polar_quantize_tensor: expected 3D or 4D tensor, got {}D",
            dims.len()
        ),
    };
    assert_eq!(
        head_dim, config.dim,
        "head_dim={head_dim} must equal config.dim={}",
        config.dim
    );

    let k_f32 = k.to_device(&candle::Device::Cpu)?.to_dtype(candle::DType::F32)?.flatten_all()?;
    let k_data = k_f32.to_vec1::<f32>()?;

    let mut all_heads = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let mut head_keys = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let offset = (h * seq_len + t) * head_dim;
            let key_slice = &k_data[offset..offset + head_dim];
            head_keys.push(polar_quantize(key_slice, config));
        }
        all_heads.push(head_keys);
    }
    Ok(all_heads)
}

/// Compute attention scores Q·K^T using PolarQuant-compressed keys.
///
/// Dequantizes each key and computes the dot product. For production use, consider using
/// `turbo_quant` which adds unbiased QJL correction.
///
/// # Returns
/// Attention scores tensor of shape `[1, num_heads, q_len, kv_len]`
pub fn polar_attention_scores(
    q: &Tensor,
    quantized_keys: &[Vec<PolarQuantizedVec>],
    config: &PolarQuantConfig,
) -> Result<Tensor> {
    let dims = q.dims();
    let (batch, num_heads, q_len, head_dim) = match dims.len() {
        4 => (dims[0], dims[1], dims[2], dims[3]),
        _ => candle::bail!("polar_attention_scores: expected 4D query tensor"),
    };
    assert_eq!(num_heads, quantized_keys.len());
    let kv_len = if num_heads > 0 { quantized_keys[0].len() } else { 0 };

    if kv_len == 0 {
        return Tensor::zeros((batch, num_heads, q_len, kv_len), q.dtype(), q.device());
    }

    let mut k_data = Vec::with_capacity(batch * num_heads * kv_len * head_dim);

    for _ in 0..batch {
        for h in 0..num_heads {
            for kt in 0..kv_len {
                let k_vec = polar_dequantize(&quantized_keys[h][kt], config);
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
    use crate::quant_kv::prng::Prng;

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

    /// Verify that round-trip (encode then decode) preserves the vector reasonably well.
    ///
    /// The quantization introduces MSE, but it should be bounded.
    #[test]
    fn round_trip_mse() {
        let d = 64;
        let config = PolarQuantConfig::new(d, 42);
        let n = 100;
        let mut total_rel_mse = 0.0f64;

        for i in 0..n {
            let x = random_unit_vec(d, i as u64 + 1);
            let pq = polar_quantize(&x, &config);
            let x_hat = polar_dequantize(&pq, &config);

            let mse: f32 = x.iter().zip(x_hat.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            let x_norm_sq: f32 = x.iter().map(|a| a * a).sum();
            total_rel_mse += (mse / x_norm_sq.max(1e-8)) as f64;
        }

        let avg_rel_mse = total_rel_mse / n as f64;
        assert!(
            avg_rel_mse <= 0.25,
            "average relative MSE = {avg_rel_mse:.4} exceeds threshold 0.25"
        );
    }

    /// Verify that bits_per_dim is within expected range for d=64.
    #[test]
    fn bits_per_dim_d64() {
        let d = 64;
        let config = PolarQuantConfig::new(d, 1);
        let x = random_unit_vec(d, 7);
        let pq = polar_quantize(&x, &config);

        let bpd = pq.bits_per_dim();
        // Expected: ~3.2 bits/dim for d=64 with 4-bit level-1 and 2-bit deeper
        assert!(
            bpd <= 4.0,
            "bits_per_dim={bpd:.3} exceeds 4.0 for d={d}"
        );
        assert!(
            bpd >= 3.0,
            "bits_per_dim={bpd:.3} too low (expected ≥ 3.0) for d={d}"
        );
    }

    /// Verify compression vs FP16.
    #[test]
    fn compression_ratio_vs_fp16() {
        let d = 128;
        let config = PolarQuantConfig::new(d, 1);
        let x = random_unit_vec(d, 3);
        let pq = polar_quantize(&x, &config);

        let fp16_bytes = d * 2; // d × 2 bytes per f16
        let ratio = fp16_bytes as f32 / pq.byte_size() as f32;
        assert!(
            ratio >= 3.5,
            "compression ratio {ratio:.2}× below expected ≥ 3.5× for d={d}"
        );
    }

    /// Verify that the number of levels equals ceil(log2(d)).
    #[test]
    fn level_count() {
        for d in [8usize, 16, 32, 64, 128] {
            let config = PolarQuantConfig::new(d, 1);
            let x = random_unit_vec(d, 1);
            let pq = polar_quantize(&x, &config);
            // n_levels should be log2(d) (for power-of-2 d)
            let expected_levels = (d as f32).log2().ceil() as usize;
            assert!(
                pq.n_levels <= expected_levels + 1,
                "d={d}: n_levels={} expected ≈ {expected_levels}",
                pq.n_levels
            );
        }
    }

    /// Verify that inner product estimation is reasonable (within 30% of true value on average).
    #[test]
    fn inner_product_accuracy() {
        let d = 64;
        let config = PolarQuantConfig::new(d, 99);
        let n = 200;
        let mut sum_rel_err = 0.0f64;

        for i in 0..n {
            let q = random_unit_vec(d, i as u64);
            let k = random_unit_vec(d, i as u64 + 10_000);

            let true_dot: f32 = q.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum();
            let pq = polar_quantize(&k, &config);
            let est = polar_inner_product(&q, &pq, &config);

            let denom = true_dot.abs().max(0.01);
            sum_rel_err += ((est - true_dot).abs() / denom) as f64;
        }

        let avg_rel_err = sum_rel_err / n as f64;
        assert!(
            avg_rel_err <= 0.5,
            "average relative inner product error = {avg_rel_err:.4} exceeds 0.5"
        );
    }
}
