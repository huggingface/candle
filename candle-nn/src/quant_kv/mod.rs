//! Quantized KV Cache: QJL, PolarQuant, and TurboQuant compression algorithms.
//!
//! This module implements three state-of-the-art KV cache quantization algorithms from
//! Google DeepMind research, enabling 4–14× memory reduction for large language model
//! inference with minimal or no accuracy loss.
//!
//! ## Algorithms
//!
//! | Algorithm | Bits/Dim | Compression | Quality | Paper |
//! |-----------|----------|-------------|---------|-------|
//! | QJL | 1.125 | 14× vs FP16 | Unbiased estimator | arxiv:2406.03482 |
//! | PolarQuant | 3.22 | 5× vs FP16 | High (LongBench: 48.37 vs 48.63) | arxiv:2502.02617 |
//! | TurboQuant | 3.5 | 4.6× vs FP16 | Full precision (LongBench: 50.06 = 50.06) | arxiv:2504.19874 |
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use candle_nn::quant_kv::{QuantAlgorithm, QuantizedKvCache};
//! use candle_nn::quant_kv::turbo_quant::TurboQuantConfig;
//!
//! // Create a quantized KV cache for 32 attention heads, head_dim=64
//! let config = TurboQuantConfig::new_3p5bit(64, /*seed=*/42);
//! let mut cache = QuantizedKvCache::new(
//!     QuantAlgorithm::TurboQuant(config),
//!     /*seq_dim=*/2,
//!     /*num_heads=*/32,
//! );
//!
//! // During autoregressive decode (one token at a time):
//! cache.append(&new_key, &new_value)?;
//! let attn_scores = cache.attention_scores(&query)?;  // [batch, heads, q_len, kv_len]
//! let v_full = cache.v()?.unwrap();                    // full-precision V
//! let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)?;
//! let output = attn_weights.matmul(&v_full.contiguous()?)?;
//! ```
//!
//! ## Module Structure
//!
//! - [`prng`]: Seeded xorshift64 PRNG + Box-Muller normal sampler (no external deps)
//! - [`fwht`]: Fast Walsh-Hadamard Transform + SRHT preconditioning
//! - [`codebook`]: Lloyd-Max scalar codebooks for Beta distribution
//! - [`qjl`]: QJL 1-bit key quantization
//! - [`polar_quant`]: PolarQuant recursive polar decomposition
//! - [`turbo_quant`]: TurboQuant two-stage hybrid
//! - [`kv_cache`]: `QuantizedKvCache` high-level interface

pub mod codebook;
pub mod fwht;
pub mod kv_cache;
pub mod polar_quant;
pub mod prng;
pub mod qjl;
pub mod turbo_quant;

pub use kv_cache::{QuantAlgorithm, QuantizedKvCache};
