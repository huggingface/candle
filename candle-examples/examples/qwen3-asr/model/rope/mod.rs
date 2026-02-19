//! Rotary embedding (RoPE/mRoPE) utilities.
//!
//! This module is largely lifted from `qwen3-tts-rs` and adapted for Qwen3-ASR.

use candle::{DType, Result, Tensor};

pub mod core;
pub mod mrope;
pub mod scaling;
pub mod standard;

/// Rotate half of the hidden dimensions.
pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(candle::D::Minus1)?;
    let half = last_dim / 2;
    let x1 = x.narrow(candle::D::Minus1, 0, half)?;
    let x2 = x.narrow(candle::D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], candle::D::Minus1)
}

/// Apply standard rotary position embedding to query and key tensors.
///
/// Expects half-dim cos/sin (head_dim/2) as produced by `standard::RotaryEmbedding`.
pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let q_embed = candle_nn::rotary_emb::rope(&q, cos, sin)?;
    let k_embed = candle_nn::rotary_emb::rope(&k, cos, sin)?;
    Ok((q_embed, k_embed))
}

/// Apply rotary position embedding manually with full-dim cos/sin.
///
/// This is used by modules that precompute full-dim cos/sin and need a manual
/// application path. Uses F32 internally for precision.
pub fn apply_rotary_pos_emb_manual(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let original_dtype = q.dtype();

    let q_f32 = q.to_dtype(DType::F32)?;
    let k_f32 = k.to_dtype(DType::F32)?;
    let cos_f32 = cos.to_dtype(DType::F32)?.unsqueeze(1)?;
    let sin_f32 = sin.to_dtype(DType::F32)?.unsqueeze(1)?;

    let q_embed = q_f32
        .broadcast_mul(&cos_f32)?
        .broadcast_add(&rotate_half(&q_f32)?.broadcast_mul(&sin_f32)?)?;
    let k_embed = k_f32
        .broadcast_mul(&cos_f32)?
        .broadcast_add(&rotate_half(&k_f32)?.broadcast_mul(&sin_f32)?)?;

    Ok((
        q_embed.to_dtype(original_dtype)?,
        k_embed.to_dtype(original_dtype)?,
    ))
}
