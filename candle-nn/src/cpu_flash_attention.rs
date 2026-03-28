//! Backward compatibility shim for CPU flash attention.
//!
//! **Deprecated:** Use `candle_nn::attention::{flash_attn, AttnMask}` instead.

use candle::{Result, Tensor, WithDType};
use std::iter::Sum;

/// Deprecated: use `candle_nn::attention::flash_attn` with `AttnMask` instead.
///
/// This shim routes through the new dispatcher which handles both B=1 and B>1.
#[deprecated(
    since = "0.9.2",
    note = "Use `candle_nn::attention::{flash_attn, AttnMask}` instead"
)]
pub fn run_flash_attn_cpu<T>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    softmax_scale: f32,
    max_bias: Option<f32>,
    softcap: Option<f32>,
) -> Result<Tensor>
where
    T: WithDType + Sum + num_traits::real::Real + 'static,
{
    use crate::attention::{flash_attn, AttnMask};

    let attn_mask = match mask {
        Some(m) => AttnMask::Mask(m.clone()),
        None => AttnMask::None,
    };
    flash_attn::<T>(q, k, v, softmax_scale, attn_mask, max_bias, softcap)
}
