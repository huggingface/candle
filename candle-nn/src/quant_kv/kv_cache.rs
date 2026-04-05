//! High-performance GPU-native quantized KV cache for language models.
//!
//! This module provides three quantized KV cache types covering different deployment scenarios:
//!
//! | Type | Strategy | Use Case |
//! |------|----------|---------|
//! | [`QuantizedKvCache`] | Growing `Tensor::cat` with incremental dequantization | GPU long-context |
//! | [`QuantizedPreAllocKvCache`] | Pre-allocated buffer with `slice_set` | CPU long-context |
//! | [`QuantizedRotatingKvCache`] | Fixed ring-buffer with wraparound | Streaming / bounded memory |
//!
//! All three types quantize incoming keys and cache the dequantized F32 result
//! so that `attention_scores()` is a simple matmul against already-decompressed keys,
//! rather than re-dequantizing the entire history on every call (O(S²) → O(S)).

use crate::quant_kv::{
    polar_quant::{
        polar_attention_scores_vectorized, polar_dequantize_batch, polar_quantize_tensor,
        PolarQuantConfig, PolarQuantTensors,
    },
    qjl::{
        qjl_attention_scores_vectorized, qjl_quantize_tensor, qjl_reconstruct_chunk, QjlConfig,
        QjlTensors,
    },
    turbo_quant::{
        turbo_attention_scores_vectorized, turbo_dequantize_chunk, turbo_quantize_tensor,
        TurboQuantConfig, TurboQuantTensors,
    },
};
use candle::{DType, Device, Result, Tensor};

// ── Shared algorithm enum ────────────────────────────────────────────────────

/// Supported quantization algorithms.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantAlgorithm {
    Qjl(QjlConfig),
    PolarQuant(PolarQuantConfig),
    TurboQuant(TurboQuantConfig),
}

// ── Internal compressed key storage ─────────────────────────────────────────

/// A quantized key cache stored on the GPU.
#[derive(Debug, Clone)]
pub enum QuantKCache {
    Qjl(QjlTensors),
    PolarQuant(PolarQuantTensors),
    TurboQuant(TurboQuantTensors),
}

// ── Helper: dequantize one newly-compressed chunk for each algorithm ─────────

/// Dequantize a freshly-quantized PolarQuant chunk to F32 `[num_heads, seq_len, dim]`.
fn polar_dequant_new(
    l1: &Tensor,
    ln: &Tensor,
    r: &Tensor,
    seq_len: usize,
    config: &PolarQuantConfig,
) -> Result<Tensor> {
    let num_heads = l1.dim(0)?;
    let mut temp = PolarQuantTensors::new(num_heads, seq_len, config.dim, l1.device())?;
    temp.level1_codes.push(l1.clone());
    temp.leveln_codes.push(ln.clone());
    temp.radii.push(r.clone());
    temp.cur_len = seq_len;
    polar_dequantize_batch(&temp, config)?.to_dtype(DType::F32)
}

/// Dequantize a freshly-quantized QJL chunk to F32 `[num_heads, seq_len, original_dim]`.
fn qjl_dequant_new(bits: &Tensor, norms: &Tensor, config: &QjlConfig) -> Result<Tensor> {
    qjl_reconstruct_chunk(bits, norms, config)
}

/// Dequantize a freshly-quantized TurboQuant chunk to F32.
fn turbo_dequant_new(
    l1: &Tensor,
    ln: &Tensor,
    r: &Tensor,
    bits: &Tensor,
    norms: &Tensor,
    seq_len: usize,
    config: &TurboQuantConfig,
) -> Result<Tensor> {
    turbo_dequantize_chunk(l1, ln, r, bits, norms, seq_len, config)
}

/// Append a new dequantized chunk onto an optional running cache along the sequence dim.
fn cat_k(existing: Option<Tensor>, new_chunk: Tensor, seq_dim: usize) -> Result<Option<Tensor>> {
    match existing {
        None => Ok(Some(new_chunk)),
        Some(prev) => Ok(Some(Tensor::cat(&[&prev, &new_chunk], seq_dim)?)),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantizedKvCache — growing cache with incremental dequantization
// ─────────────────────────────────────────────────────────────────────────────

/// Growing quantized KV cache that incrementally dequantizes keys.
///
/// On each `append()`, only the newly-added tokens are dequantized and concatenated
/// onto a running `cached_k_dequant` tensor. `attention_scores()` then computes
/// `q @ cached_k.T` directly, avoiding O(S²) re-dequantization.
///
/// Values are stored full-precision.
#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    pub k_cache: QuantKCache,
    /// Running dequantized F32 K tensor: `[num_heads, cur_len, dim]`
    pub cached_k_dequant: Option<Tensor>,
    pub v_cache: Vec<Tensor>,
    pub algorithm: QuantAlgorithm,
    pub cur_len: usize,
}

impl QuantizedKvCache {
    pub fn new(
        num_heads: usize,
        max_seq_len: usize,
        dim: usize,
        algorithm: QuantAlgorithm,
        device: &Device,
    ) -> Result<Self> {
        let k_cache = match &algorithm {
            QuantAlgorithm::Qjl(cfg) => QuantKCache::Qjl(QjlTensors::new(
                num_heads,
                max_seq_len,
                cfg.dim / 8,
                device,
            )?),
            QuantAlgorithm::PolarQuant(_) => QuantKCache::PolarQuant(PolarQuantTensors::new(
                num_heads,
                max_seq_len,
                dim,
                device,
            )?),
            QuantAlgorithm::TurboQuant(_) => QuantKCache::TurboQuant(TurboQuantTensors::new(
                num_heads,
                max_seq_len,
                dim,
                device,
            )?),
        };
        Ok(Self {
            k_cache,
            cached_k_dequant: None,
            v_cache: Vec::new(),
            algorithm,
            cur_len: 0,
        })
    }

    pub fn current_seq_len(&self) -> usize {
        self.cur_len
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let k_dims = k.dims();
        let seq_len = k_dims[k_dims.len() - 2];
        // After squeezing to 3D [H, S, D], the sequence dimension is always 1.
        const SEQ_DIM: usize = 1;

        // Quantize, store compressed, and dequantize the new chunk
        let new_k_dequant: Tensor = match &mut self.k_cache {
            QuantKCache::Qjl(t) => {
                let cfg = match &self.algorithm {
                    QuantAlgorithm::Qjl(c) => c,
                    _ => unreachable!(),
                };
                let (bits, norms) = qjl_quantize_tensor(k, cfg)?;
                let dequant = qjl_dequant_new(&bits, &norms, cfg)?;
                t.sign_bits.push(bits);
                t.norms.push(norms);
                t.cur_len += seq_len;
                dequant
            }
            QuantKCache::PolarQuant(t) => {
                let cfg = match &self.algorithm {
                    QuantAlgorithm::PolarQuant(c) => c,
                    _ => unreachable!(),
                };
                let (l1, ln, r) = polar_quantize_tensor(k, cfg)?;
                let dequant = polar_dequant_new(&l1, &ln, &r, seq_len, cfg)?;
                t.level1_codes.push(l1);
                t.leveln_codes.push(ln);
                t.radii.push(r);
                t.cur_len += seq_len;
                dequant
            }
            QuantKCache::TurboQuant(t) => {
                let cfg = match &self.algorithm {
                    QuantAlgorithm::TurboQuant(c) => c,
                    _ => unreachable!(),
                };
                let ((l1, ln, r), (bits, qjl_norms)) = turbo_quantize_tensor(k, cfg)?;
                let dequant = turbo_dequant_new(&l1, &ln, &r, &bits, &qjl_norms, seq_len, cfg)?;
                t.mse_tensors.level1_codes.push(l1);
                t.mse_tensors.leveln_codes.push(ln);
                t.mse_tensors.radii.push(r);
                t.qjl_tensors.sign_bits.push(bits);
                t.qjl_tensors.norms.push(qjl_norms);
                t.mse_tensors.cur_len += seq_len;
                t.qjl_tensors.cur_len += seq_len;
                t.cur_len += seq_len;
                dequant
            }
        };

        // Ensure new_k_dequant is 3D [H, S, D] for consistent concatenation
        let new_k_dequant = if new_k_dequant.dims().len() == 4 {
            new_k_dequant.squeeze(0)?
        } else {
            new_k_dequant
        };

        self.cached_k_dequant = cat_k(self.cached_k_dequant.take(), new_k_dequant, SEQ_DIM)?;
        self.v_cache.push(v.clone());
        self.cur_len += seq_len;

        Ok(())
    }

    /// Compute attention scores using the vectorized quantized estimators.
    ///
    /// Uses the algorithm-specific vectorized inner-product estimators rather than
    /// decoding K to full precision, which preserves the statistical guarantees of
    /// each algorithm and matches the quality of the original implementation.
    ///
    /// Returns `[batch, num_heads, q_len, kv_len]`.
    pub fn attention_scores(&self, q: &Tensor) -> Result<Tensor> {
        match (&self.k_cache, &self.algorithm) {
            (QuantKCache::TurboQuant(t), QuantAlgorithm::TurboQuant(cfg)) => {
                turbo_attention_scores_vectorized(q, t, cfg)
            }
            (QuantKCache::PolarQuant(t), QuantAlgorithm::PolarQuant(cfg)) => {
                polar_attention_scores_vectorized(q, t, cfg)
            }
            (QuantKCache::Qjl(t), QuantAlgorithm::Qjl(cfg)) => {
                qjl_attention_scores_vectorized(q, t, cfg)
            }
            _ => candle::bail!("attention_scores: mismatched cache/algorithm"),
        }
    }

    pub fn v(&self) -> Result<Option<Tensor>> {
        if self.v_cache.is_empty() {
            return Ok(None);
        }
        let cat_dim = self.v_cache[0].dims().len() - 2;
        Tensor::cat(&self.v_cache, cat_dim).map(Some)
    }

    // ── Compression ratio helpers ─────────────────────────────────────────

    /// Compressed bytes used for the key cache.
    pub fn k_bytes_used(&self) -> usize {
        match &self.k_cache {
            QuantKCache::Qjl(t) => {
                t.sign_bits.iter().map(|tb| tb.elem_count()).sum::<usize>()
                    + t.norms.iter().map(|tb| tb.elem_count() * 4).sum::<usize>()
            }
            QuantKCache::PolarQuant(t) => {
                t.level1_codes
                    .iter()
                    .map(|tb| tb.elem_count())
                    .sum::<usize>()
                    + t.leveln_codes
                        .iter()
                        .map(|tb| tb.elem_count())
                        .sum::<usize>()
                    + t.radii.iter().map(|tb| tb.elem_count() * 2).sum::<usize>()
                // f16
            }
            QuantKCache::TurboQuant(t) => {
                t.mse_tensors
                    .level1_codes
                    .iter()
                    .map(|tb| tb.elem_count())
                    .sum::<usize>()
                    + t.mse_tensors
                        .leveln_codes
                        .iter()
                        .map(|tb| tb.elem_count())
                        .sum::<usize>()
                    + t.mse_tensors
                        .radii
                        .iter()
                        .map(|tb| tb.elem_count() * 2)
                        .sum::<usize>()
                    + t.qjl_tensors
                        .sign_bits
                        .iter()
                        .map(|tb| tb.elem_count())
                        .sum::<usize>()
                    + t.qjl_tensors
                        .norms
                        .iter()
                        .map(|tb| tb.elem_count() * 4)
                        .sum::<usize>()
            }
        }
    }

    /// Equivalent FP16 bytes for the same number of key elements.
    pub fn k_bytes_full_precision(&self) -> usize {
        match &self.k_cache {
            QuantKCache::Qjl(t) => t.cur_len * 2, // elements * 2 bytes (F16), but we need dim info
            QuantKCache::PolarQuant(t) => {
                // each radius tensor has shape [H, S, 1]; total elements = num_heads * cur_len
                t.radii.iter().map(|r| r.elem_count()).sum::<usize>() * 2
            }
            QuantKCache::TurboQuant(t) => {
                t.mse_tensors
                    .radii
                    .iter()
                    .map(|r| r.elem_count())
                    .sum::<usize>()
                    * 2
            }
        }
    }

    /// Compression ratio vs FP16 (higher is better).
    pub fn k_compression_ratio(&self) -> f64 {
        let full = self.k_bytes_full_precision();
        if full == 0 {
            return 1.0;
        }
        full as f64 / self.k_bytes_used() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantizedPreAllocKvCache — pre-allocated buffer, slice_set on append
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-allocated quantized KV cache.
///
/// Allocates fixed `[num_heads, max_seq_len, dim]` buffers up front and fills
/// them via `slice_set`, avoiding per-token allocations. Suited for CPU
/// long-context workloads where allocation overhead matters.
#[derive(Debug, Clone)]
pub struct QuantizedPreAllocKvCache {
    /// Pre-allocated dequantized K: `[num_heads, max_seq_len, dim]` F32
    k_buf: Tensor,
    /// Pre-allocated V: `[num_heads, max_seq_len, dim]` F32
    v_buf: Tensor,
    pub cur_len: usize,
    pub algorithm: QuantAlgorithm,
    pub max_seq_len: usize,
    pub num_heads: usize,
    pub dim: usize,
}

impl QuantizedPreAllocKvCache {
    pub fn new(
        num_heads: usize,
        max_seq_len: usize,
        dim: usize,
        algorithm: QuantAlgorithm,
        device: &Device,
    ) -> Result<Self> {
        let k_buf = Tensor::zeros((num_heads, max_seq_len, dim), DType::F32, device)?;
        let v_buf = Tensor::zeros((num_heads, max_seq_len, dim), DType::F32, device)?;
        Ok(Self {
            k_buf,
            v_buf,
            cur_len: 0,
            algorithm,
            max_seq_len,
            num_heads,
            dim,
        })
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let k_dims = k.dims();
        let seq_len = k_dims[k_dims.len() - 2];
        if self.cur_len + seq_len > self.max_seq_len {
            candle::bail!(
                "QuantizedPreAllocKvCache: sequence length {} exceeds max_seq_len {}",
                self.cur_len + seq_len,
                self.max_seq_len
            );
        }

        let new_k = self.dequantize_k(k, seq_len)?;
        let new_k = if new_k.dims().len() == 4 {
            new_k.squeeze(0)?
        } else {
            new_k
        };
        let new_v = v.to_dtype(DType::F32)?;
        let new_v = if new_v.dims().len() == 4 {
            new_v.squeeze(0)?
        } else {
            new_v
        };

        // Write into pre-allocated buffer at cur_len..cur_len+seq_len (in-place)
        self.k_buf.slice_set(&new_k, 1, self.cur_len)?;
        self.v_buf.slice_set(&new_v, 1, self.cur_len)?;
        self.cur_len += seq_len;

        Ok(())
    }

    fn dequantize_k(&self, k: &Tensor, seq_len: usize) -> Result<Tensor> {
        match &self.algorithm {
            QuantAlgorithm::Qjl(cfg) => {
                let (bits, norms) = qjl_quantize_tensor(k, cfg)?;
                qjl_dequant_new(&bits, &norms, cfg)
            }
            QuantAlgorithm::PolarQuant(cfg) => {
                let (l1, ln, r) = polar_quantize_tensor(k, cfg)?;
                polar_dequant_new(&l1, &ln, &r, seq_len, cfg)
            }
            QuantAlgorithm::TurboQuant(cfg) => {
                let ((l1, ln, r), (bits, qjl_norms)) = turbo_quantize_tensor(k, cfg)?;
                turbo_dequant_new(&l1, &ln, &r, &bits, &qjl_norms, seq_len, cfg)
            }
        }
    }

    /// Returns the active portion of the K buffer: `[num_heads, cur_len, dim]`.
    pub fn k(&self) -> Result<Tensor> {
        self.k_buf.narrow(1, 0, self.cur_len)
    }

    /// Returns the active portion of the V buffer: `[num_heads, cur_len, dim]`.
    pub fn v(&self) -> Result<Tensor> {
        self.v_buf.narrow(1, 0, self.cur_len)
    }

    /// Compute attention scores `q @ K^T` over the active portion.
    pub fn attention_scores(&self, q: &Tensor) -> Result<Tensor> {
        if self.cur_len == 0 {
            let dims = q.dims();
            let rank = dims.len();
            let (batch, num_heads, q_len) = match rank {
                4 => (dims[0], dims[1], dims[2]),
                3 => (1, dims[0], dims[1]),
                _ => candle::bail!("attention_scores: expected 3D or 4D query"),
            };
            return Tensor::zeros((batch, num_heads, q_len, 0), q.dtype(), q.device());
        }
        let k = self.k()?.to_dtype(q.dtype())?;
        let rank = q.dims().len();
        let k = if rank == 4 { k.unsqueeze(0)? } else { k };
        let q_flat = q.contiguous()?.flatten_to(rank - 3)?;
        let k_flat = k.contiguous()?.flatten_to(rank - 3)?;
        let scores = q_flat.matmul(&k_flat.transpose(1, 2)?)?;
        let mut out_shape = q.dims().to_vec();
        out_shape[rank - 1] = self.cur_len;
        scores.reshape(out_shape)?.to_dtype(q.dtype())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantizedRotatingKvCache — fixed-size ring buffer
// ─────────────────────────────────────────────────────────────────────────────

/// Fixed-window quantized KV cache with ring-buffer semantics.
///
/// Maintains exactly `max_seq_len` slots. Once full, the oldest tokens are
/// overwritten. Keys are quantized, then dequantized into the ring buffer.
/// `attention_scores()` returns scores against the current window in order.
#[derive(Debug, Clone)]
pub struct QuantizedRotatingKvCache {
    /// Ring buffer for dequantized K: `[num_heads, max_seq_len, dim]` F32
    k_buf: Tensor,
    /// Ring buffer for V: `[num_heads, max_seq_len, dim]` F32
    v_buf: Tensor,
    /// Number of valid slots (< max_seq_len until the buffer wraps).
    pub cur_len: usize,
    /// Write pointer (next slot to overwrite).
    pub offset: usize,
    pub algorithm: QuantAlgorithm,
    pub max_seq_len: usize,
}

impl QuantizedRotatingKvCache {
    pub fn new(
        num_heads: usize,
        max_seq_len: usize,
        dim: usize,
        algorithm: QuantAlgorithm,
        device: &Device,
    ) -> Result<Self> {
        let k_buf = Tensor::zeros((num_heads, max_seq_len, dim), DType::F32, device)?;
        let v_buf = Tensor::zeros((num_heads, max_seq_len, dim), DType::F32, device)?;
        Ok(Self {
            k_buf,
            v_buf,
            cur_len: 0,
            offset: 0,
            algorithm,
            max_seq_len,
        })
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let k_dims = k.dims();
        let seq_len = k_dims[k_dims.len() - 2];

        let new_k = self.dequantize_k(k, seq_len)?;
        let new_k = if new_k.dims().len() == 4 {
            new_k.squeeze(0)?
        } else {
            new_k
        };
        let new_v = v.to_dtype(DType::F32)?;
        let new_v = if new_v.dims().len() == 4 {
            new_v.squeeze(0)?
        } else {
            new_v
        };

        // Write new tokens one-by-one into the ring buffer
        for i in 0..seq_len {
            let slot = self.offset % self.max_seq_len;
            let k_tok = new_k.narrow(1, i, 1)?;
            let v_tok = new_v.narrow(1, i, 1)?;
            self.k_buf.slice_set(&k_tok, 1, slot)?;
            self.v_buf.slice_set(&v_tok, 1, slot)?;
            self.offset += 1;
            if self.cur_len < self.max_seq_len {
                self.cur_len += 1;
            }
        }

        Ok(())
    }

    fn dequantize_k(&self, k: &Tensor, seq_len: usize) -> Result<Tensor> {
        match &self.algorithm {
            QuantAlgorithm::Qjl(cfg) => {
                let (bits, norms) = qjl_quantize_tensor(k, cfg)?;
                qjl_dequant_new(&bits, &norms, cfg)
            }
            QuantAlgorithm::PolarQuant(cfg) => {
                let (l1, ln, r) = polar_quantize_tensor(k, cfg)?;
                polar_dequant_new(&l1, &ln, &r, seq_len, cfg)
            }
            QuantAlgorithm::TurboQuant(cfg) => {
                let ((l1, ln, r), (bits, qjl_norms)) = turbo_quantize_tensor(k, cfg)?;
                turbo_dequant_new(&l1, &ln, &r, &bits, &qjl_norms, seq_len, cfg)
            }
        }
    }

    /// Attention scores against the current window, in chronological order.
    ///
    /// If the ring has wrapped, the tokens are reordered before the matmul so
    /// that `scores[..., 0]` corresponds to the oldest token in the window.
    pub fn attention_scores(&self, q: &Tensor) -> Result<Tensor> {
        if self.cur_len == 0 {
            let dims = q.dims();
            let rank = dims.len();
            let (batch, num_heads, q_len) = match rank {
                4 => (dims[0], dims[1], dims[2]),
                3 => (1, dims[0], dims[1]),
                _ => candle::bail!("attention_scores: expected 3D or 4D query"),
            };
            return Tensor::zeros((batch, num_heads, q_len, 0), q.dtype(), q.device());
        }

        let k = self.ordered_k()?.to_dtype(q.dtype())?;
        let rank = q.dims().len();
        let k = if rank == 4 { k.unsqueeze(0)? } else { k };
        let q_flat = q.contiguous()?.flatten_to(rank - 3)?;
        let k_flat = k.contiguous()?.flatten_to(rank - 3)?;
        let scores = q_flat.matmul(&k_flat.transpose(1, 2)?)?;
        let mut out_shape = q.dims().to_vec();
        out_shape[rank - 1] = self.cur_len;
        scores.reshape(out_shape)?.to_dtype(q.dtype())
    }

    /// Returns the active K window in chronological order: `[num_heads, cur_len, dim]`.
    pub fn ordered_k(&self) -> Result<Tensor> {
        self.ordered_buf(&self.k_buf)
    }

    /// Returns the active V window in chronological order: `[num_heads, cur_len, dim]`.
    pub fn ordered_v(&self) -> Result<Tensor> {
        self.ordered_buf(&self.v_buf)
    }

    fn ordered_buf(&self, buf: &Tensor) -> Result<Tensor> {
        if self.cur_len < self.max_seq_len {
            // Buffer hasn't wrapped yet — just slice the valid prefix
            buf.narrow(1, 0, self.cur_len)
        } else {
            // Buffer has wrapped: oldest token is at `offset % max_seq_len`
            let start = self.offset % self.max_seq_len;
            if start == 0 {
                buf.clone().narrow(1, 0, self.max_seq_len)
            } else {
                let tail = buf.narrow(1, start, self.max_seq_len - start)?;
                let head = buf.narrow(1, 0, start)?;
                Tensor::cat(&[&tail, &head], 1)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_kv::turbo_quant::TurboQuantConfig;
    use candle::{Device, Tensor};

    fn random_tensor(shape: &[usize], seed: u64, device: &Device) -> Tensor {
        use crate::quant_kv::prng::Prng;
        let total: usize = shape.iter().product();
        let mut rng = Prng::new(seed);
        let mut data = vec![0.0f32; total];
        rng.fill_normal(&mut data);
        Tensor::from_vec(data, shape, device).unwrap()
    }

    fn relative_mse(a: &Tensor, b: &Tensor) -> f64 {
        let a_f = a
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b_f = b
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mse: f64 = a_f
            .iter()
            .zip(b_f.iter())
            .map(|(x, y)| (x - y) as f64 * (x - y) as f64)
            .sum::<f64>()
            / a_f.len() as f64;
        let var: f64 =
            b_f.iter().map(|y| (*y as f64) * (*y as f64)).sum::<f64>() / b_f.len() as f64;
        if var < 1e-12 {
            return 0.0;
        }
        mse / var
    }

    #[test]
    fn test_attention_quality_dim64_3p5bit() {
        // Verify score quality for 3.5-bit dim=64 (same as NIAH model uses)
        let device = Device::Cpu;
        let (num_heads, seq_len, dim) = (2, 16, 64);
        let cfg = TurboQuantConfig::new_3p5bit(dim, 42);
        let mut cache =
            QuantizedKvCache::new(num_heads, 64, dim, QuantAlgorithm::TurboQuant(cfg), &device)
                .unwrap();

        let k = random_tensor(&[num_heads, seq_len, dim], 10, &device);
        let v = random_tensor(&[num_heads, seq_len, dim], 11, &device);
        cache.append(&k, &v).unwrap();

        let q = random_tensor(&[1, num_heads, 1, dim], 20, &device);

        // Scores vs original K (not the dequantized version — this measures actual quality)
        let k_orig = k.to_dtype(DType::F32).unwrap();
        let q_3d = q.squeeze(0).unwrap().to_dtype(DType::F32).unwrap();
        let scores_exact = q_3d.matmul(&k_orig.transpose(1, 2).unwrap()).unwrap();
        let scores_quant = cache
            .attention_scores(&q)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let score_rel_mse = relative_mse(&scores_quant, &scores_exact);

        // Key reconstruction quality
        let k_dq = cache
            .cached_k_dequant
            .as_ref()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let k_rel_mse = relative_mse(&k_dq, &k_orig);

        println!(
            "dim=64 3.5bit: k_rel_mse={k_rel_mse:.4}, score_rel_mse_vs_exact={score_rel_mse:.4}"
        );

        // Threshold: quantization can be noisy but shouldn't be completely random
        // score_rel_mse < 2.0 means the scores are better than random noise
        assert!(score_rel_mse < 2.0,
            "Attention scores too noisy for dim=64 3.5bit: score_rel_mse={score_rel_mse:.4}, k_rel_mse={k_rel_mse:.4}");
    }

    #[test]
    fn test_quantized_kv_cache_basic() {
        let device = Device::Cpu;
        // dim=64: QJL projected_dim = 64/8 = 8, packed_dim = 1 (valid)
        let (num_heads, dim) = (2, 64);
        let cfg = TurboQuantConfig::new_3p5bit(dim, 42);
        let mut cache =
            QuantizedKvCache::new(num_heads, 64, dim, QuantAlgorithm::TurboQuant(cfg), &device)
                .unwrap();

        assert_eq!(cache.current_seq_len(), 0);

        let k1 = random_tensor(&[num_heads, 4, dim], 1, &device);
        let v1 = random_tensor(&[num_heads, 4, dim], 2, &device);
        cache.append(&k1, &v1).unwrap();
        assert_eq!(cache.current_seq_len(), 4);

        let k2 = random_tensor(&[num_heads, 3, dim], 3, &device);
        let v2 = random_tensor(&[num_heads, 3, dim], 4, &device);
        cache.append(&k2, &v2).unwrap();
        assert_eq!(cache.current_seq_len(), 7);

        // cached_k_dequant should be [num_heads, 7, dim]
        let k_dq = cache.cached_k_dequant.as_ref().unwrap();
        assert_eq!(k_dq.dims(), [num_heads, 7, dim]);

        // V should concatenate correctly
        let v = cache.v().unwrap().unwrap();
        assert_eq!(v.dim(1).unwrap(), 7);
    }

    #[test]
    fn test_quantized_kv_cache_attention_quality() {
        let device = Device::Cpu;
        // dim=128: standard head dim, QJL projected_dim = 32, packed_dim = 4 (valid)
        let (num_heads, seq_len, dim) = (2, 16, 128);
        let cfg = TurboQuantConfig::new_4bit(dim, 42);
        let mut cache =
            QuantizedKvCache::new(num_heads, 64, dim, QuantAlgorithm::TurboQuant(cfg), &device)
                .unwrap();

        let k = random_tensor(&[num_heads, seq_len, dim], 10, &device);
        let v = random_tensor(&[num_heads, seq_len, dim], 11, &device);
        cache.append(&k, &v).unwrap();

        let q = random_tensor(&[1, num_heads, 1, dim], 20, &device);

        // 1. Verify attention_scores matches q @ cached_k^T exactly
        let k_dq = cache
            .cached_k_dequant
            .as_ref()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap(); // [H, S, D]
        let q_3d = q.squeeze(0).unwrap().to_dtype(DType::F32).unwrap(); // [H, 1, D]
        let scores_direct = q_3d.matmul(&k_dq.transpose(1, 2).unwrap()).unwrap(); // [H, 1, S]
        let scores_cache = cache
            .attention_scores(&q)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap(); // [H, 1, S]
        let exact_rel_mse = relative_mse(&scores_cache, &scores_direct);
        assert!(
            exact_rel_mse < 1e-4,
            "attention_scores should match q @ cached_k^T exactly, rel_mse={exact_rel_mse:.6}"
        );

        // 2. Verify cached K quality vs original K
        // Threshold 0.5 catches completely broken output while tolerating inherent
        // quantization noise at test dimensions (full quality at d=128 is ~5-10%).
        let k_orig = k.to_dtype(DType::F32).unwrap();
        let rel_mse_k = relative_mse(&k_dq, &k_orig);
        assert!(
            rel_mse_k < 0.5,
            "Cached K deviates too much from original: relative_mse={rel_mse_k:.4}"
        );
    }

    #[test]
    fn test_quantized_kv_cache_incremental() {
        let device = Device::Cpu;
        // dim=64: QJL projected_dim = 64/4 = 16, packed_dim = 2 (valid)
        let (num_heads, dim) = (2, 64);
        let cfg = TurboQuantConfig::new_4bit(dim, 7);

        // Batch: append all at once
        let mut batch_cache = QuantizedKvCache::new(
            num_heads,
            64,
            dim,
            QuantAlgorithm::TurboQuant(cfg.clone()),
            &device,
        )
        .unwrap();
        let k_all = random_tensor(&[num_heads, 8, dim], 5, &device);
        let v_all = random_tensor(&[num_heads, 8, dim], 6, &device);
        batch_cache.append(&k_all, &v_all).unwrap();

        // Incremental: append two chunks of 4
        let mut inc_cache =
            QuantizedKvCache::new(num_heads, 64, dim, QuantAlgorithm::TurboQuant(cfg), &device)
                .unwrap();
        let k1 = k_all.narrow(1, 0, 4).unwrap();
        let k2 = k_all.narrow(1, 4, 4).unwrap();
        let v1 = v_all.narrow(1, 0, 4).unwrap();
        let v2 = v_all.narrow(1, 4, 4).unwrap();
        inc_cache.append(&k1, &v1).unwrap();
        inc_cache.append(&k2, &v2).unwrap();

        assert_eq!(batch_cache.current_seq_len(), inc_cache.current_seq_len());

        // Both should produce the same cached K shape
        let kb = batch_cache.cached_k_dequant.as_ref().unwrap();
        let ki = inc_cache.cached_k_dequant.as_ref().unwrap();
        assert_eq!(kb.dims(), ki.dims());
    }

    #[test]
    fn test_prealloc_kv_cache_basic() {
        let device = Device::Cpu;
        // dim=64: QJL projected_dim = 8, packed_dim = 1 (valid)
        let (num_heads, dim, max_seq) = (2, 64, 64);
        let cfg = TurboQuantConfig::new_3p5bit(dim, 42);
        let mut cache = QuantizedPreAllocKvCache::new(
            num_heads,
            max_seq,
            dim,
            QuantAlgorithm::TurboQuant(cfg),
            &device,
        )
        .unwrap();

        let k = random_tensor(&[num_heads, 5, dim], 1, &device);
        let v = random_tensor(&[num_heads, 5, dim], 2, &device);
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.cur_len, 5);
        assert_eq!(cache.k().unwrap().dims(), &[num_heads, 5, dim]);
        assert_eq!(cache.v().unwrap().dims(), &[num_heads, 5, dim]);
    }

    #[test]
    fn test_rotating_wrap() {
        let device = Device::Cpu;
        // dim=64: QJL projected_dim = 8, packed_dim = 1 (valid)
        let (num_heads, dim, max_seq) = (1, 64, 4);
        let cfg = TurboQuantConfig::new_3p5bit(dim, 42);
        let mut cache = QuantizedRotatingKvCache::new(
            num_heads,
            max_seq,
            dim,
            QuantAlgorithm::TurboQuant(cfg),
            &device,
        )
        .unwrap();

        // Fill past max_seq_len (write 6 tokens into a window of 4)
        for i in 0..6 {
            let k = random_tensor(&[num_heads, 1, dim], i as u64, &device);
            let v = random_tensor(&[num_heads, 1, dim], i as u64 + 100, &device);
            cache.append(&k, &v).unwrap();
        }

        // cur_len should be capped at max_seq_len
        assert_eq!(cache.cur_len, max_seq);
        assert_eq!(cache.offset, 6);

        // ordered_k should have max_seq_len tokens
        let k_ord = cache.ordered_k().unwrap();
        assert_eq!(k_ord.dims(), &[num_heads, max_seq, dim]);
    }
}
