//! Attention implementations for Candle.
//!
//! ```ignore
//! use candle_nn::attention::{flash_attn, AttnMask};
//!
//! // Causal (optimized loop-bound path)
//! let out = flash_attn::<f32>(&q, &k, &v, scale, AttnMask::causal(), None, None)?;
//!
//! // With KV cache offset
//! let out = flash_attn::<f32>(&q, &k, &v, scale, AttnMask::causal_with_offset(cache_len), None, None)?;
//!
//! // Custom mask tensor
//! let out = flash_attn::<f32>(&q, &k, &v, scale, AttnMask::Mask(mask), None, None)?;
//! ```

pub mod cpu_flash;

use candle::Tensor;

pub use cpu_flash::flash_attn;

/// Attention mask specification.
///
/// - `None`: Full bidirectional attention
/// - `Causal`: Loop-bound masking (skips ~50% of work, no tensor allocation)
/// - `Mask`: Explicit tensor for arbitrary patterns (sliding window, block-sparse)
#[derive(Debug, Clone, Default)]
pub enum AttnMask {
    #[default]
    None,
    Causal {
        kv_offset: usize,
    },
    Mask(Tensor),
}

impl AttnMask {
    #[inline]
    pub fn causal() -> Self {
        AttnMask::Causal { kv_offset: 0 }
    }

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
