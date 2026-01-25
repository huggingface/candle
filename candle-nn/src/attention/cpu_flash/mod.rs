//! CPU flash attention implementations.
//!
//! - `standard`: General-purpose with explicit mask tensor
//! - `causal`: Optimized loop-bound causal masking (no tensor allocation)

pub mod causal;
pub mod standard;

use candle::{Result, Tensor, WithDType};
use std::iter::Sum;

use super::AttnMask;

/// Flash attention with automatic dispatch.
///
/// Selects optimal implementation:
/// - `AttnMask::Causal` → `causal.rs` (loop-bound, no mask tensor)
/// - `AttnMask::None` or `AttnMask::Mask` → `standard.rs`
///
/// # Arguments
/// * `q` - Query tensor, shape `(B, S, H, D)`
/// * `k` - Key tensor, shape `(B, KV_S, KV_H, D)`
/// * `v` - Value tensor, shape `(B, KV_S, KV_H, D)`
/// * `softmax_scale` - Scale factor (typically `1/sqrt(head_dim)`)
/// * `attn_mask` - Masking strategy
/// * `max_bias` - ALiBi max bias (`None` to disable)
/// * `softcap` - Logit soft-capping (`None` to disable)
///
/// # Returns
/// Output tensor with shape `(B, H, S, D)`
pub fn flash_attn<T>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    attn_mask: AttnMask,
    max_bias: Option<f32>,
    softcap: Option<f32>,
) -> Result<Tensor>
where
    T: WithDType + Sum + num_traits::real::Real,
{
    match attn_mask {
        AttnMask::Causal { kv_offset } => {
            // Optimized path: loop-bound causal masking
            causal::run_causal_attn_cpu::<T>(q, k, v, softmax_scale, kv_offset, max_bias, softcap)
        }
        AttnMask::None => {
            // No masking
            standard::run_flash_attn_cpu::<T>(q, k, v, None, softmax_scale, max_bias, softcap)
        }
        AttnMask::Mask(mask) => {
            // Explicit mask tensor
            standard::run_flash_attn_cpu::<T>(
                q,
                k,
                v,
                Some(&mask),
                softmax_scale,
                max_bias,
                softcap,
            )
        }
    }
}
