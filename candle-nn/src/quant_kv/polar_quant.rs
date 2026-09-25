//! PolarQuant: A highly accurate quantization method for KV caches.
//!
//! ## Paper Reference
//!
//! "PolarQuant: Leveraging Polar Transformation for Efficient Key Cache Quantization and Decoding Acceleration"
//! (2025) — arxiv:2502.00527

use super::codebook::{get_polar_codebook, Codebook};
use candle::{Result, Tensor};
use half::f16;

/// Configuration for PolarQuant quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct PolarQuantConfig {
    pub dim: usize,
    pub seed: u64,
    pub bits_level1: u32,
    pub bits_deep: u32,
    pub codebooks: Vec<Codebook>,
}

impl PolarQuantConfig {
    pub fn new(dim: usize, seed: u64, bits_level1: u32, bits_deep: u32) -> Self {
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

    pub fn num_levels(&self) -> usize {
        self.codebooks.len()
    }

    pub fn codebook_for_level(&self, level: usize) -> &Codebook {
        &self.codebooks[level - 1]
    }
}

/// A PolarQuant-compressed vector.
#[derive(Debug, Clone)]
pub struct PolarQuantizedVec {
    pub level1_codes: Vec<u8>,
    pub leveln_codes: Vec<u8>,
    pub final_radius: f16,
    pub n_levels: usize,
    pub dim: usize,
    pub bits_level1: u32,
    pub bits_deep: u32,
    pub level_counts: Vec<usize>,
}

/// GPU-native storage for a batch of PolarQuant-quantized keys.
#[derive(Debug, Clone)]
pub struct PolarQuantTensors {
    pub level1_codes: Vec<Tensor>,
    pub leveln_codes: Vec<Tensor>,
    pub radii: Vec<Tensor>,
    pub cur_len: usize,
}

impl PolarQuantTensors {
    pub fn new(
        _num_heads: usize,
        _max_seq_len: usize,
        _dim: usize,
        _device: &candle::Device,
    ) -> Result<Self> {
        Ok(Self {
            level1_codes: Vec::new(),
            leveln_codes: Vec::new(),
            radii: Vec::new(),
            cur_len: 0,
        })
    }

    pub fn cat_level1(&self) -> Result<Tensor> {
        let cat_dim = self.level1_codes[0].dims().len() - 2;
        Tensor::cat(&self.level1_codes, cat_dim)
    }

    pub fn cat_leveln(&self) -> Result<Tensor> {
        let cat_dim = self.leveln_codes[0].dims().len() - 2;
        Tensor::cat(&self.leveln_codes, cat_dim)
    }

    pub fn cat_radii(&self) -> Result<Tensor> {
        let cat_dim = self.radii[0].dims().len() - 2;
        Tensor::cat(&self.radii, cat_dim)
    }
}

/// Quantize a key tensor using PolarQuant.
pub fn polar_quantize_tensor(
    k: &Tensor,
    config: &PolarQuantConfig,
) -> Result<(Tensor, Tensor, Tensor)> {
    let device = k.device();
    let dims = k.dims();
    let (num_heads, seq_len, d) = match dims.len() {
        4 => (dims[1], dims[2], dims[3]),
        3 => (dims[0], dims[1], dims[2]),
        _ => candle::bail!("polar_quantize_tensor: expected 3D or 4D tensor"),
    };

    let cb_l1 = config.codebook_for_level(1);
    let k_data = k
        .to_dtype(candle::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let l1_p = d.div_ceil(2).div_ceil(2);
    let ln_p = d.div_ceil(8);
    let n_levels = config.num_levels();
    let bits_deep = config.bits_deep as usize;

    // Precompute bit offset for each deep level in ln_codes.
    // Layout: level 2 bits first, then level 3, ..., then level n_levels.
    let mut ln_bit_offsets = vec![0usize; n_levels + 1];
    {
        let mut cumul = 0usize;
        let mut n_angles = d / 2; // level-1 has d/2 angles; level-2 has d/4, etc.
        for offset in ln_bit_offsets.iter_mut().skip(2).take(n_levels - 1) {
            n_angles /= 2;
            *offset = cumul;
            cumul += n_angles * bits_deep;
        }
    }

    let mut l1_vec = Vec::with_capacity(num_heads * seq_len * l1_p);
    let mut ln_vec = vec![0u8; num_heads * seq_len * ln_p]; // zero-initialized
    let mut radii_vec = Vec::with_capacity(num_heads * seq_len);

    let mut rotated = vec![0.0f32; d];

    for idx in 0..(num_heads * seq_len) {
        let start = idx * d;

        // Compute radius from original k (SRHT is norm-preserving for power-of-2 d)
        let mut norm_sq = 0.0f32;
        for i in 0..d {
            norm_sq += k_data[start + i] * k_data[start + i];
        }
        radii_vec.push(norm_sq.sqrt());

        // Apply SRHT so angle distribution matches codebook assumptions (uniform / Beta)
        super::fwht::srht_precondition(&k_data[start..start + d], config.seed, &mut rotated);

        // Level 1: compute angles and level-1 magnitudes
        let mut current_mags: Vec<f32> = Vec::with_capacity(d / 2);
        for i in 0..(d / 2) {
            let v1 = rotated[2 * i];
            let v2 = rotated[2 * i + 1];
            let angle = v2.atan2(v1);
            let code = cb_l1.quantize(angle);
            // Pack two 4-bit codes per byte
            if i % 2 == 1 {
                let prev = l1_vec.pop().unwrap();
                l1_vec.push(prev | (code << 4));
            } else {
                l1_vec.push(code);
            }
            current_mags.push((v1 * v1 + v2 * v2).sqrt());
        }

        // Deep levels 2..n_levels: recursively quantize pairs of magnitudes
        let ln_base = idx * ln_p; // byte offset for this vector's ln_codes
        for (level_idx, &bit_base) in ln_bit_offsets[2..=n_levels].iter().enumerate() {
            let level = level_idx + 2;
            let cb_deep = config.codebook_for_level(level);
            let n_pairs = current_mags.len() / 2;
            let mut next_mags = Vec::with_capacity(n_pairs);

            for (j, pair) in current_mags.chunks_exact(2).enumerate() {
                let (a, b) = (pair[0], pair[1]);
                // atan2(b, a) ∈ [0, π/2] for non-negative a, b — matches Beta codebook
                let angle = b.atan2(a);
                let code = cb_deep.quantize(angle);

                // Pack bits_deep bits of code (LSB first)
                for bit_i in 0..bits_deep {
                    let bit = (code >> bit_i) & 1;
                    let global_bit = bit_base + j * bits_deep + bit_i;
                    let byte_idx = ln_base + global_bit / 8;
                    if bit != 0 {
                        ln_vec[byte_idx] |= 1 << (global_bit % 8);
                    }
                }

                next_mags.push(a.hypot(b));
            }
            current_mags = next_mags;
        }
        // current_mags now has 1 element (the L2 norm, already stored in radii_vec)
    }

    let l1 = Tensor::from_vec(l1_vec, (num_heads, seq_len, l1_p), device)?;
    let ln = Tensor::from_vec(ln_vec, (num_heads, seq_len, ln_p), device)?;
    let r = Tensor::from_vec(radii_vec, (num_heads, seq_len, 1), device)?;

    Ok((l1, ln, r))
}

/// Dequantize a batch of keys on the GPU using Rayon fallback.
pub fn polar_dequantize_batch(
    k_tensors: &PolarQuantTensors,
    config: &PolarQuantConfig,
) -> Result<Tensor> {
    use rayon::prelude::*;

    let l1_tensor = k_tensors.cat_level1()?;
    let ln_tensor = k_tensors.cat_leveln()?;
    let radii_tensor = k_tensors.cat_radii()?;

    let device = l1_tensor.device();
    let num_heads = l1_tensor.dim(0)?;
    let kv_len = k_tensors.cur_len;
    let d = config.dim;

    let l1_data = l1_tensor.flatten_all()?.to_vec1::<u8>()?;
    let ln_data = ln_tensor.flatten_all()?.to_vec1::<u8>()?;
    let radii_data = radii_tensor.flatten_all()?.to_vec1::<f32>()?;

    let l1_stride = l1_tensor.dim(2)?;
    let ln_stride = ln_tensor.dim(2)?;

    // Precompute bit offsets for deep levels (same layout as encoding).
    // ln_bit_offsets[lvl] = first bit index in ln_data for level lvl (lvl >= 2).
    let n_levels = config.num_levels();
    let bits_deep = config.bits_deep as usize;
    let mut ln_bit_offsets = vec![0usize; n_levels + 1];
    {
        let mut cumul = 0usize;
        let mut n_angles = d / 2;
        for offset in ln_bit_offsets.iter_mut().skip(2).take(n_levels - 1) {
            n_angles /= 2;
            *offset = cumul;
            cumul += n_angles * bits_deep;
        }
    }

    let mut all_keys_vec = vec![0.0f32; num_heads * kv_len * d];

    let t_count = kv_len;
    let dim_d = d;
    let srht_seed = config.seed; // u64 is Copy — safe to capture by value in Rayon closure

    all_keys_vec.par_chunks_mut(dim_d).enumerate().for_each(
        |(idx, out_vec): (usize, &mut [f32])| {
            let h = idx / t_count;
            let t = idx % t_count;

            let l1_start = (h * t_count + t) * l1_stride;
            let ln_base = (h * t_count + t) * ln_stride;
            let r = radii_data[h * t_count + t];
            let cb = config.codebook_for_level(1);

            // Recover level-1 pair magnitudes by descending the deep-level tree.
            // Start with the total radius and split level-by-level using the stored angles.
            // This reconstructs the correct energy distribution across pairs.
            // NOTE: Assumes d is a power of 2 (true for all standard Llama/Mistral head dims).
            let mut current_mags: Vec<f32> = vec![r];
            for level in (2..=n_levels).rev() {
                let cb_deep = config.codebook_for_level(level);
                let n_parents = current_mags.len();
                let bit_base = ln_bit_offsets[level];
                let mut next_mags = Vec::with_capacity(n_parents * 2);

                for (j, &parent) in current_mags.iter().enumerate() {
                    let mut code = 0u8;
                    for bit_i in 0..bits_deep {
                        let global_bit = bit_base + j * bits_deep + bit_i;
                        let byte_idx = ln_base + global_bit / 8;
                        let bit = (ln_data[byte_idx] >> (global_bit % 8)) & 1;
                        code |= bit << bit_i;
                    }
                    let angle = cb_deep.dequantize(code);
                    next_mags.push(parent * angle.cos());
                    next_mags.push(parent * angle.sin());
                }
                current_mags = next_mags;
            }
            // current_mags now holds the d/2 level-1 pair magnitudes.

            // Reconstruct in SRHT-rotated space using recovered magnitudes and level-1 angles.
            let mut rotated_reconstruction = vec![0.0f32; dim_d];
            for i in 0..(dim_d / 2) {
                let byte = l1_data[l1_start + (i / 2)];
                let code = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
                let angle = cb.dequantize(code);
                let m_i = current_mags[i];
                rotated_reconstruction[2 * i] = m_i * angle.cos();
                rotated_reconstruction[2 * i + 1] = m_i * angle.sin();
            }

            // Invert SRHT to recover vectors in original key space
            super::fwht::srht_inverse(&rotated_reconstruction, srht_seed, out_vec);
        },
    );

    Tensor::from_vec(all_keys_vec, (num_heads, kv_len, d), device)
}

/// Vectorized attention scores for PolarQuant.
pub fn polar_attention_scores_vectorized(
    q: &Tensor,
    k_tensors: &PolarQuantTensors,
    config: &PolarQuantConfig,
) -> Result<Tensor> {
    let device = q.device();
    let dims = q.dims();
    let (_batch, _num_heads, _q_len, _d) = match dims.len() {
        4 => (dims[0], dims[1], dims[2], dims[3]),
        3 => (1, dims[0], dims[1], dims[2]),
        _ => candle::bail!("polar_attention_scores_vectorized: expected 3D or 4D tensor"),
    };
    let kv_len = k_tensors.cur_len;

    if kv_len == 0 {
        return Tensor::zeros((_batch, _num_heads, _q_len, 0), q.dtype(), device);
    }

    let k_dequant = polar_dequantize_batch(k_tensors, config)?;
    let k_dequant = k_dequant.to_dtype(q.dtype())?;

    let rank = q.dims().len();
    // k should be [num_heads, kv_len, head_dim]
    // 2. Ensure rank matches q by unsqueezing if needed
    let k_dequant = if rank == 4 {
        k_dequant.unsqueeze(0)?
    } else {
        k_dequant
    };

    let original_q_shape = q.shape().clone();

    let q_reshape = q.contiguous()?.flatten_to(rank - 3)?;
    let k_reshape = k_dequant.contiguous()?.flatten_to(rank - 3)?;

    let scores = q_reshape.matmul(&k_reshape.transpose(1, 2)?)?;
    let mut scores_shape = original_q_shape.dims().to_vec();
    scores_shape[rank - 1] = k_dequant.dim(rank - 2)?;
    scores.reshape(scores_shape)?.to_dtype(q.dtype())
}
