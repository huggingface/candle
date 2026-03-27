//! CPU flash attention implementations.
//!
//! - `standard`: General-purpose with explicit mask tensor (B,S,H,D)
//! - `causal`: Optimized loop-bound causal masking (B,S,H,D), best for B=1
//! - `varlen`: Packed variable-length sequences (total_q, H, D), best for B>1
//!
//! The top-level [`flash_attn`] function automatically dispatches:
//! - **B=1**: single-batch optimized paths in `standard`/`causal`
//! - **B>1** (f32/f16, no softcap): packs into varlen format for better throughput
//! - **Fallback**: multi-batch `standard`/`causal` paths for unsupported configs

pub mod causal;
pub mod standard;
pub mod varlen;

use candle::{DType, Result, Tensor, WithDType};
use std::iter::Sum;

use super::AttnMask;

/// Flash attention with automatic dispatch.
///
/// Selects optimal implementation based on batch size, mask type, and dtype:
/// - **B=1**: uses single-batch optimized kernels (direct slice access, no batch overhead)
/// - **B>1 + Causal/None + f32/f16**: packs to varlen format (avoids batch-dim stride overhead)
/// - **Explicit mask or unsupported dtype**: falls back to general-purpose batched kernel
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
    let b = q.dims()[0];

    // B>1: must go through packed varlen path — the B=1 kernels hard-error on B>1.
    if b > 1 {
        let dt = q.dtype();
        let varlen_ok = (dt == DType::F32 || dt == DType::F16) && softcap.is_none();
        let mask_ok = matches!(&attn_mask, AttnMask::Causal { .. } | AttnMask::None);

        if !varlen_ok || !mask_ok {
            candle::bail!(
                "CPU flash attention with B>1 requires: f32/f16 dtype, no softcap, \
                 and Causal or None mask. Got B={b}, dtype={dt:?}, softcap={softcap:?}, \
                 mask={}",
                match &attn_mask {
                    AttnMask::Causal { .. } => "Causal",
                    AttnMask::None => "None",
                    AttnMask::Mask(_) => "Mask(tensor)",
                }
            );
        }

        return flash_attn_via_varlen(q, k, v, softmax_scale, &attn_mask, max_bias);
    }

    // B=1: dedicated single-batch kernels (no batch indexing, direct slices)
    match attn_mask {
        AttnMask::Causal { kv_offset } => {
            causal::run_causal_attn_cpu::<T>(q, k, v, softmax_scale, kv_offset, max_bias, softcap)
        }
        AttnMask::None => {
            standard::run_flash_attn_cpu::<T>(q, k, v, None, softmax_scale, max_bias, softcap)
        }
        AttnMask::Mask(mask) => standard::run_flash_attn_cpu::<T>(
            q,
            k,
            v,
            Some(&mask),
            softmax_scale,
            max_bias,
            softcap,
        ),
    }
}

/// Reshape batched (B,S,H,D) tensors into packed varlen format and dispatch.
///
/// Returns output in (B, H, S, D) to match the standard `flash_attn` contract.
fn flash_attn_via_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    attn_mask: &AttnMask,
    max_bias: Option<f32>,
) -> Result<Tensor> {
    let q_dims = q.dims();
    let k_dims = k.dims();
    let (b, s_q, h_q, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let (s_kv, h_kv) = (k_dims[1], k_dims[2]);

    let causal = attn_mask.is_causal();

    // Pack (B, S, H, D) → (B*S, H, D)
    let q_packed = q.contiguous()?.reshape((b * s_q, h_q, d))?;
    let k_packed = k.contiguous()?.reshape((b * s_kv, h_kv, d))?;
    let v_packed = v.contiguous()?.reshape((b * s_kv, h_kv, d))?;

    // Build uniform seqlens
    let device = q.device();
    let seqlens_q = Tensor::from_vec(vec![s_q as u32; b], b, device)?;
    let seqlens_k = Tensor::from_vec(vec![s_kv as u32; b], b, device)?;

    // ALiBi: convert max_bias to per-head slopes tensor
    let alibi_slopes = if let Some(mb) = max_bias {
        if mb > 0.0 {
            let n2 = 2_usize.pow((h_q as f32).log2().ceil() as u32);
            let slopes: Vec<f32> = (0..h_q)
                .map(|h| 2.0f32.powf(-mb * ((h + 1) as f32) / n2 as f32))
                .collect();
            Some(Tensor::from_vec(slopes, h_q, device)?)
        } else {
            None
        }
    } else {
        None
    };

    let ctx = varlen::flash_attn_varlen_cpu(
        &q_packed,
        &k_packed,
        &v_packed,
        alibi_slopes.as_ref(),
        &seqlens_q,
        &seqlens_k,
        s_q,
        s_kv,
        softmax_scale,
        causal,
        None,
        None,
    )?;

    // ctx: (B*S_q, H_q, D) → (B, S_q, H_q, D) → (B, H_q, S_q, D)
    ctx.reshape((b, s_q, h_q, d))?.transpose(1, 2)?.contiguous()
}
