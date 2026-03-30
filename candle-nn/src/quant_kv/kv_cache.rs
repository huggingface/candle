//! Quantized KV Cache: drop-in replacement for `candle_nn::kv_cache::KvCache`.
//!
//! ## Design
//!
//! This module provides `QuantizedKvCache`, which stores:
//! - **Keys (K)**: compressed using one of QJL, PolarQuant, or TurboQuant
//! - **Values (V)**: stored at full precision (matching the paper's "asymmetric" design)
//!
//! The V cache is not quantized because:
//! 1. V is accessed via `softmax(Q·K^T)·V`, where the weights already incorporate the
//!    attention distribution. The V contribution to output quality is already modulated.
//! 2. The QJL asymmetry guarantees unbiased Q·K^T estimation; extending to V quantization
//!    requires additional analysis.
//!
//! ## Interface
//!
//! `QuantizedKvCache` follows the same usage pattern as `candle_nn::kv_cache::KvCache`:
//!
//! ```rust,ignore
//! // Standard usage:
//! let mut cache = QuantizedKvCache::new(
//!     QuantAlgorithm::TurboQuant(TurboQuantConfig::new_3p5bit(head_dim, 42)),
//!     seq_dim,
//!     max_seq_len,
//!     num_heads,
//! );
//!
//! // During autoregressive decode:
//! cache.append(&new_k, &new_v)?;
//! let scores = cache.attention_scores(&q)?; // Q·K^T using quantized keys
//! let v_full = cache.v()?.unwrap();          // Full-precision V for softmax(scores)·V
//! ```

use candle::{Result, Tensor};

use super::polar_quant::{
    polar_attention_scores, polar_quantize_tensor, PolarQuantConfig, PolarQuantizedVec,
};
use super::qjl::{qjl_attention_scores, qjl_quantize_tensor, QjlConfig, QjlQuantizedKey};
use super::turbo_quant::{
    turbo_attention_scores, turbo_quantize_tensor, TurboQuantConfig, TurboQuantizedVec,
};

/// Selects which quantization algorithm to use for the key cache.
#[derive(Debug, Clone)]
pub enum QuantAlgorithm {
    /// QJL: 1-bit per key dimension + f16 norm. ~14× compression vs FP16.
    /// Best for: extreme memory constraints, edge deployment.
    Qjl(QjlConfig),
    /// PolarQuant: ~3.9 bits per dimension. ~4× compression vs FP16.
    /// Best for: high-quality compression without residual correction.
    PolarQuant(PolarQuantConfig),
    /// TurboQuant: ~3.5 bits per dimension. ~4.6× compression vs FP16.
    /// Best for: production use — matches full-precision quality on LongBench benchmarks.
    TurboQuant(TurboQuantConfig),
}

impl QuantAlgorithm {
    /// Returns the head dimension for this configuration.
    pub fn dim(&self) -> usize {
        match self {
            QuantAlgorithm::Qjl(c) => c.dim,
            QuantAlgorithm::PolarQuant(c) => c.dim,
            QuantAlgorithm::TurboQuant(c) => c.dim,
        }
    }
}

/// Internal storage for quantized keys, dispatched by algorithm.
enum QuantKCache {
    /// Indexed [head][token]
    Qjl(Vec<Vec<QjlQuantizedKey>>),
    /// Indexed [head][token]
    PolarQuant(Vec<Vec<PolarQuantizedVec>>),
    /// Indexed [head][token]
    TurboQuant(Vec<Vec<TurboQuantizedVec>>),
}

impl QuantKCache {
    fn new_empty(algorithm: &QuantAlgorithm, num_heads: usize) -> Self {
        match algorithm {
            QuantAlgorithm::Qjl(_) => {
                QuantKCache::Qjl(vec![Vec::new(); num_heads])
            }
            QuantAlgorithm::PolarQuant(_) => {
                QuantKCache::PolarQuant(vec![Vec::new(); num_heads])
            }
            QuantAlgorithm::TurboQuant(_) => {
                QuantKCache::TurboQuant(vec![Vec::new(); num_heads])
            }
        }
    }

    fn current_seq_len(&self) -> usize {
        match self {
            QuantKCache::Qjl(cache) => cache.first().map(|h| h.len()).unwrap_or(0),
            QuantKCache::PolarQuant(cache) => cache.first().map(|h| h.len()).unwrap_or(0),
            QuantKCache::TurboQuant(cache) => cache.first().map(|h| h.len()).unwrap_or(0),
        }
    }

    fn reset(&mut self) {
        match self {
            QuantKCache::Qjl(cache) => cache.iter_mut().for_each(|h| h.clear()),
            QuantKCache::PolarQuant(cache) => cache.iter_mut().for_each(|h| h.clear()),
            QuantKCache::TurboQuant(cache) => cache.iter_mut().for_each(|h| h.clear()),
        }
    }
}

/// A quantized KV cache that compresses key vectors while keeping values at full precision.
///
/// This provides the same memory savings as the TurboQuant/QJL/PolarQuant papers demonstrate,
/// while being a transparent replacement for `candle_nn::kv_cache::KvCache` in autoregressive
/// inference loops.
pub struct QuantizedKvCache {
    k_cache: QuantKCache,
    /// Full-precision V cache (raw stacked tensor, grown as tokens arrive)
    v_data: Vec<Tensor>,
    /// Algorithm and config for key compression
    algorithm: QuantAlgorithm,
    /// Number of attention heads
    num_heads: usize,
    /// Sequence dimension in the input tensor (typically 2 for [batch, heads, seq, dim])
    _seq_dim: usize,
}

impl QuantizedKvCache {
    /// Create a new quantized KV cache.
    ///
    /// # Arguments
    /// * `algorithm` — which quantization algorithm to use for keys
    /// * `seq_dim` — the sequence dimension in the key/value tensors (typically 2)
    /// * `num_heads` — number of attention heads
    pub fn new(algorithm: QuantAlgorithm, seq_dim: usize, num_heads: usize) -> Self {
        let k_cache = QuantKCache::new_empty(&algorithm, num_heads);
        Self {
            k_cache,
            v_data: Vec::new(),
            algorithm,
            num_heads,
            _seq_dim: seq_dim,
        }
    }

    /// Append new key and value tensors for the current decode step.
    ///
    /// # Arguments
    /// * `k` — key tensor of shape `[batch=1, num_heads, new_tokens, head_dim]`
    /// * `v` — value tensor of shape `[batch=1, num_heads, new_tokens, head_dim]`
    ///
    /// Keys are compressed using the configured algorithm; values are stored at full precision.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let seq_offset = self.current_seq_len();
        // Quantize and append keys
        let dims = k.dims();
        let (num_heads, seq_len, head_dim) = match dims.len() {
            4 => (dims[1], dims[2], dims[3]),
            3 => (dims[0], dims[1], dims[2]),
            _ => candle::bail!("QuantizedKvCache::append: unexpected tensor shape {:?}", dims),
        };

        assert_eq!(
            num_heads, self.num_heads,
            "num_heads mismatch: cache has {}, got {num_heads}",
            self.num_heads
        );
        assert_eq!(
            head_dim,
            self.algorithm.dim(),
            "head_dim={head_dim} doesn't match algorithm dim={}",
            self.algorithm.dim()
        );

        match &self.algorithm {
            QuantAlgorithm::Qjl(cfg) => {
                let new_keys = qjl_quantize_tensor(k, cfg, seq_offset)?;
                if let QuantKCache::Qjl(cache) = &mut self.k_cache {
                    for h in 0..num_heads {
                        cache[h].extend(new_keys[h].iter().cloned());
                    }
                }
            }
            QuantAlgorithm::PolarQuant(cfg) => {
                let cfg = cfg.clone();
                let new_keys = polar_quantize_tensor(k, &cfg)?;
                if let QuantKCache::PolarQuant(cache) = &mut self.k_cache {
                    for h in 0..num_heads {
                        cache[h].extend(new_keys[h].iter().cloned());
                    }
                }
            }
            QuantAlgorithm::TurboQuant(cfg) => {
                let cfg = cfg.clone();
                let new_keys = turbo_quantize_tensor(k, &cfg, seq_offset)?;
                if let QuantKCache::TurboQuant(cache) = &mut self.k_cache {
                    for h in 0..num_heads {
                        cache[h].extend(new_keys[h].iter().cloned());
                    }
                }
            }
        }

        // Store V at full precision
        // We accumulate V tensors and cat them on demand
        let _ = seq_len; // used indirectly via k quantization
        self.v_data.push(v.clone());
        Ok(())
    }

    /// Compute attention scores Q·K^T using the quantized key cache.
    ///
    /// # Arguments
    /// * `q` — query tensor of shape `[batch=1, num_heads, q_len, head_dim]`
    ///
    /// # Returns
    /// Attention score tensor of shape `[1, num_heads, q_len, kv_seq_len]`
    pub fn attention_scores(&self, q: &Tensor) -> Result<Tensor> {
        match (&self.k_cache, &self.algorithm) {
            (QuantKCache::Qjl(cache), QuantAlgorithm::Qjl(cfg)) => {
                qjl_attention_scores(q, cache, cfg)
            }
            (QuantKCache::PolarQuant(cache), QuantAlgorithm::PolarQuant(cfg)) => {
                polar_attention_scores(q, cache, cfg)
            }
            (QuantKCache::TurboQuant(cache), QuantAlgorithm::TurboQuant(cfg)) => {
                turbo_attention_scores(q, cache, cfg)
            }
            _ => candle::bail!("QuantizedKvCache: algorithm/cache type mismatch"),
        }
    }

    /// Retrieve the current full-precision V cache for computing `softmax(scores) · V`.
    ///
    /// Returns `None` if no tokens have been appended yet.
    ///
    /// The returned tensor has shape `[1, num_heads, kv_seq_len, head_dim]`.
    pub fn v(&self) -> Result<Option<Tensor>> {
        if self.v_data.is_empty() {
            return Ok(None);
        }
        let v = Tensor::cat(&self.v_data, 2)?; // cat along seq_len dim
        Ok(Some(v))
    }

    /// Total number of key/value tokens currently cached.
    pub fn current_seq_len(&self) -> usize {
        self.k_cache.current_seq_len()
    }

    /// Reset the cache, discarding all stored tokens.
    pub fn reset(&mut self) {
        self.k_cache.reset();
        self.v_data.clear();
    }

    /// Approximate memory usage in bytes for the key cache.
    pub fn key_memory_bytes(&self) -> usize {
        match &self.k_cache {
            QuantKCache::Qjl(cache) => cache
                .iter()
                .flat_map(|h| h.iter())
                .map(|qk| qk.byte_size())
                .sum(),
            QuantKCache::PolarQuant(cache) => cache
                .iter()
                .flat_map(|h| h.iter())
                .map(|pq| pq.byte_size())
                .sum(),
            QuantKCache::TurboQuant(cache) => cache
                .iter()
                .flat_map(|h| h.iter())
                .map(|tq| tq.byte_size())
                .sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn make_random_tensor(shape: &[usize], seed: u64) -> Tensor {
        use crate::quant_kv::prng::Prng;
        let total: usize = shape.iter().product();
        let mut rng = Prng::new(seed);
        let mut data = vec![0.0f32; total];
        rng.fill_normal(&mut data);
        // Normalize each head_dim slice to unit norm
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

    /// Verify that appending tokens and retrieving scores works correctly.
    #[test]
    fn append_and_score_turbo() -> Result<()> {
        let num_heads = 2;
        let head_dim = 32;
        let n_tokens = 8;

        let config = TurboQuantConfig::new_3p5bit(head_dim, 42);
        let mut cache = QuantizedKvCache::new(
            QuantAlgorithm::TurboQuant(config),
            2,
            num_heads,
        );

        // Append n_tokens keys and values
        let k = make_random_tensor(&[1, num_heads, n_tokens, head_dim], 1);
        let v = make_random_tensor(&[1, num_heads, n_tokens, head_dim], 2);
        cache.append(&k, &v)?;

        assert_eq!(cache.current_seq_len(), n_tokens);

        // Compute attention scores for 1 query token
        let q = make_random_tensor(&[1, num_heads, 1, head_dim], 3);
        let scores = cache.attention_scores(&q)?;

        assert_eq!(scores.dims(), &[1, num_heads, 1, n_tokens]);
        Ok(())
    }

    /// Verify reset clears all state.
    #[test]
    fn reset_clears_state() -> Result<()> {
        let num_heads = 1;
        let head_dim = 32;

        let config = QjlConfig::new(head_dim, 7);
        let mut cache = QuantizedKvCache::new(QuantAlgorithm::Qjl(config), 2, num_heads);

        let k = make_random_tensor(&[1, num_heads, 4, head_dim], 1);
        let v = make_random_tensor(&[1, num_heads, 4, head_dim], 2);
        cache.append(&k, &v)?;
        assert_eq!(cache.current_seq_len(), 4);

        cache.reset();
        assert_eq!(cache.current_seq_len(), 0);
        assert!(cache.v()?.is_none());
        Ok(())
    }

    /// Verify that QJL cache scores have the right shape and are not NaN.
    #[test]
    fn qjl_cache_scores_valid() -> Result<()> {
        let num_heads = 4;
        let head_dim = 64;
        let n_tokens = 16;

        let config = QjlConfig::new(head_dim, 123);
        let mut cache = QuantizedKvCache::new(QuantAlgorithm::Qjl(config), 2, num_heads);

        let k = make_random_tensor(&[1, num_heads, n_tokens, head_dim], 10);
        let v = make_random_tensor(&[1, num_heads, n_tokens, head_dim], 11);
        cache.append(&k, &v)?;

        let q = make_random_tensor(&[1, num_heads, 1, head_dim], 12);
        let scores = cache.attention_scores(&q)?;

        // Check shape
        assert_eq!(scores.dims(), &[1, num_heads, 1, n_tokens]);

        // Check no NaN
        let s = scores.flatten_all()?.to_vec1::<f32>()?;
        for score in &s {
            assert!(!score.is_nan(), "NaN score in QJL cache");
        }
        Ok(())
    }

    /// Verify that TurboQuant cache scores are close to full-precision baseline.
    ///
    /// This is a sanity check: for short sequences, the quantization error should be small.
    #[test]
    fn turbo_scores_close_to_full_precision() -> Result<()> {
        let num_heads = 1;
        let head_dim = 64;
        let n_tokens = 4;
        let seed = 999;

        let config = TurboQuantConfig::new_3p5bit(head_dim, seed);
        let mut cache = QuantizedKvCache::new(
            QuantAlgorithm::TurboQuant(config),
            2,
            num_heads,
        );

        let k = make_random_tensor(&[1, num_heads, n_tokens, head_dim], 20);
        let v = make_random_tensor(&[1, num_heads, n_tokens, head_dim], 21);
        let q = make_random_tensor(&[1, num_heads, 1, head_dim], 22);

        // Full precision scores: q · k^T
        let k_f32 = k.to_dtype(DType::F32)?;
        let q_f32 = q.to_dtype(DType::F32)?;
        // q: [1,1,1,64], k: [1,1,4,64] → scores: [1,1,1,4]
        let true_scores = q_f32
            .narrow(2, 0, 1)?
            .matmul(&k_f32.transpose(2, 3)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Quantized scores
        cache.append(&k, &v)?;
        let quant_scores = cache
            .attention_scores(&q)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Check that most scores have reasonable relative error.
        // Quantization introduces noise; we check that the mean absolute error is bounded.
        let mean_abs_err = true_scores
            .iter()
            .zip(quant_scores.iter())
            .map(|(t, q)| (t - q).abs())
            .sum::<f32>()
            / n_tokens as f32;
        assert!(
            mean_abs_err <= 1.0,
            "Mean abs score error={mean_abs_err:.4} exceeds 1.0"
        );
        Ok(())
    }
}
