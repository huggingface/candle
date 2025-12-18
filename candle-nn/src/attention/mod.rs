//! Attention implementations for Candle.
//!
//! # Usage
//!
//! ```ignore
//! use candle_nn::attention::{flash_attn, AttnMask};
//!
//! // Decode with causal mask (uses optimized generative path)
//! let out = flash_attn::<f32>(
//!     &q, &k, &v,
//!     1.0 / (head_dim as f32).sqrt(),
//!     AttnMask::causal_with_offset(kv_cache_len),
//!     None, None,
//! )?;
//! ```

pub mod cpu_flash;

use candle::Tensor;

// Re-export main API
pub use cpu_flash::flash_attn;

/// Attention mask specification.
///
/// Using an enum instead of raw tensors enables optimizations like computing
/// causal masks via loop bounds rather than materializing mask tensors.
#[derive(Debug, Clone, Copy)]
pub enum AttnMask<'a> {
    /// No masking — full bidirectional attention.
    None,

    /// Causal masking via efficient loop bounds (no tensor allocation).
    ///
    /// `kv_offset`: Number of prior KV positions when using KV cache.
    /// - Prefill: `kv_offset = 0`
    /// - Decode: `kv_offset = cached_kv_len`
    Causal { kv_offset: usize },

    /// Custom mask tensor for arbitrary patterns (sliding window, block-sparse, etc).
    ///
    /// Shape: `(B, Q_LEN, KV_LEN)` or broadcastable.
    /// Values: `0.0` to attend, `NEG_INFINITY` to mask.
    Mask(&'a Tensor),
}

impl Default for AttnMask<'_> {
    fn default() -> Self {
        AttnMask::None
    }
}

impl<'a> AttnMask<'a> {
    /// Causal mask for prefill (no KV offset).
    #[inline]
    pub fn causal() -> Self {
        AttnMask::Causal { kv_offset: 0 }
    }

    /// Causal mask for decode with KV cache.
    #[inline]
    pub fn causal_with_offset(kv_offset: usize) -> Self {
        AttnMask::Causal { kv_offset }
    }

    #[inline]
    pub fn is_causal(&self) -> bool {
        matches!(self, AttnMask::Causal { .. })
    }

    #[inline]
    pub fn kv_offset(&self) -> usize {
        match self {
            AttnMask::Causal { kv_offset } => *kv_offset,
            _ => 0,
        }
    }
}
