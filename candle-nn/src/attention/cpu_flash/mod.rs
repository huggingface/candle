//! CPU flash attention implementations.
//!
//! - `standard`: General-purpose implementation (original)
//! - `generative`: Optimized for decode (q_len=1)

pub mod generative;
pub mod standard;

use candle::{Result, Tensor, WithDType};
use std::iter::Sum;

use super::AttnMask;

/// Flash attention with automatic dispatch.
///
/// Selects optimal implementation:
/// - `q_len == 1` + `AttnMask::Causal` → generative (optimized decode)
/// - Otherwise → standard
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
    attn_mask: AttnMask<'_>,
    max_bias: Option<f32>,
    softcap: Option<f32>,
) -> Result<Tensor>
where
    T: WithDType + Sum + num_traits::real::Real,
{
    let q_len = q.dims()[1];

    eprintln!(
        ">>> flash_attn dispatch: q_len={}, is_causal={}",
        q_len,
        attn_mask.is_causal()
    ); // Add this

    // Fast path: decode with causal mask
    if q_len == 1 && attn_mask.is_causal() {
        eprintln!(">>> taking generative path");
        return generative::flash_attn_generative::<T>(
            q,
            k,
            v,
            softmax_scale,
            &attn_mask,
            max_bias.unwrap_or(0.0),
            softcap.unwrap_or(0.0),
        );
    }

    // Standard path: convert AttnMask to Option<Tensor>
    let mask_tensor = attn_mask_to_tensor(&attn_mask, q, k)?;
    standard::run_flash_attn_cpu::<T>(
        q,
        k,
        v,
        mask_tensor.as_ref(),
        softmax_scale,
        max_bias,
        softcap,
    )
}

/// Convert [`AttnMask`] to `Option<Tensor>` for the standard implementation.
fn attn_mask_to_tensor(attn_mask: &AttnMask<'_>, q: &Tensor, k: &Tensor) -> Result<Option<Tensor>> {
    match attn_mask {
        AttnMask::None => Ok(None),
        AttnMask::Mask(t) => Ok(Some((*t).clone())),
        AttnMask::Causal { kv_offset } => {
            let (b, q_len, _, _) = q.dims4()?;
            let kv_len = k.dims()[1];
            let device = q.device();
            let dtype = q.dtype();

            let mask: Vec<f32> = (0..b)
                .flat_map(|_| {
                    (0..q_len).flat_map(|q_pos| {
                        (0..kv_len).map(move |kv_pos| {
                            if kv_pos <= q_pos + kv_offset {
                                0.0
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                })
                .collect();

            let mask = Tensor::from_vec(mask, (b, q_len, kv_len), device)?.to_dtype(dtype)?;
            Ok(Some(mask))
        }
    }
}
