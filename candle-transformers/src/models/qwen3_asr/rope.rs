//! Multimodal Rotary Position Embedding (mRoPE) for Qwen3-ASR.

use candle::{DType, Device, IndexOp, Result, Tensor};

// ── Multimodal RoPE (mRoPE) ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MultimodalRotaryEmbedding {
    inv_freq: Tensor,
}

impl MultimodalRotaryEmbedding {
    pub fn new(head_dim: usize, rope_theta: f64, device: &Device) -> Result<Self> {
        if !head_dim.is_multiple_of(2) {
            candle::bail!("head_dim must be even, got {head_dim}");
        }
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half_dim, device)?;
        Ok(Self { inv_freq })
    }

    pub fn forward(&self, x: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let dtype = x.dtype();
        let inv_freq = self.inv_freq.reshape((1, 1, 1, ()))?.to_dtype(DType::F32)?;
        let position_ids = position_ids
            .unsqueeze(candle::D::Minus1)?
            .to_dtype(DType::F32)?;
        let freqs = inv_freq.broadcast_mul(&position_ids)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok((cos, sin))
    }
}

// ── mRoPE application ────────────────────────────────────────────────

pub fn apply_multimodal_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
    interleaved: bool,
    interleave_masks: Option<&[Tensor; 3]>,
) -> Result<(Tensor, Tensor)> {
    if interleaved {
        apply_mrope_interleaved(q, k, cos, sin, mrope_section, interleave_masks)
    } else {
        apply_mrope_concat(q, k, cos, sin, mrope_section)
    }
}

fn apply_mrope_concat(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor)> {
    let (_modalities, _batch, _seq_len, half_dim) = cos.dims4()?;
    let section_sum: usize = mrope_section.iter().sum();
    if section_sum != half_dim {
        candle::bail!("mrope_section sum ({section_sum}) must equal cos half_dim ({half_dim})");
    }
    let mut cos_parts: Vec<Tensor> = Vec::new();
    let mut sin_parts: Vec<Tensor> = Vec::new();
    let mut offset = 0usize;
    for (i, &section_size) in mrope_section.iter().enumerate() {
        let cos_modality = cos.i(i)?;
        let sin_modality = sin.i(i)?;
        let cos_section = cos_modality.narrow(candle::D::Minus1, offset, section_size)?;
        let sin_section = sin_modality.narrow(candle::D::Minus1, offset, section_size)?;
        cos_parts.push(cos_section);
        sin_parts.push(sin_section);
        offset = offset.saturating_add(section_size);
    }
    let cos_half =
        Tensor::cat(&cos_parts.iter().collect::<Vec<_>>(), candle::D::Minus1)?.contiguous()?;
    let sin_half =
        Tensor::cat(&sin_parts.iter().collect::<Vec<_>>(), candle::D::Minus1)?.contiguous()?;
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let q_embed = candle_nn::rotary_emb::rope(&q, &cos_half, &sin_half)?;
    let k_embed = candle_nn::rotary_emb::rope(&k, &cos_half, &sin_half)?;
    Ok((q_embed, k_embed))
}

fn apply_mrope_interleaved(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
    interleave_masks: Option<&[Tensor; 3]>,
) -> Result<(Tensor, Tensor)> {
    let (modalities, _batch, _seq_len, half_dim) = cos.dims4()?;
    let modality_num = mrope_section.len();
    if modalities < 3 {
        candle::bail!(
            "interleaved mRoPE requires at least 3 modalities, got {modalities} (mrope_section has {modality_num} entries)"
        );
    }
    if modality_num != 3 {
        candle::bail!(
            "interleaved mRoPE requires exactly 3 mrope_section entries, got {modality_num}"
        );
    }
    let original_dtype = cos.dtype();
    let cos_half = cos.contiguous()?.to_dtype(DType::F32)?;
    let sin_half = sin.contiguous()?.to_dtype(DType::F32)?;

    let cos_m0 = cos_half.i(0)?.contiguous()?;
    let sin_m0 = sin_half.i(0)?.contiguous()?;
    let cos_m1 = cos_half.i(1)?.contiguous()?;
    let sin_m1 = sin_half.i(1)?.contiguous()?;
    let cos_m2 = cos_half.i(2)?.contiguous()?;
    let sin_m2 = sin_half.i(2)?.contiguous()?;

    let (mask_m0, mask_m1, mask_m2) = match interleave_masks {
        Some([m0, m1, m2]) => (m0.clone(), m1.clone(), m2.clone()),
        None => {
            let m1_end = (mrope_section[1] * modality_num).min(half_dim);
            let m2_end = (mrope_section[2] * modality_num).min(half_dim);

            let mut mask_m0 = vec![1.0f32; half_dim];
            let mut mask_m1 = vec![0.0f32; half_dim];
            let mut mask_m2 = vec![0.0f32; half_dim];

            for pos in 0..half_dim {
                if pos >= 1 && pos < m1_end && (pos - 1) % modality_num == 0 {
                    mask_m0[pos] = 0.0;
                    mask_m1[pos] = 1.0;
                } else if pos >= 2 && pos < m2_end && (pos - 2) % modality_num == 0 {
                    mask_m0[pos] = 0.0;
                    mask_m2[pos] = 1.0;
                }
            }

            let device = cos.device();
            let mask_m0 = Tensor::from_vec(mask_m0, (1, 1, half_dim), device)?;
            let mask_m1 = Tensor::from_vec(mask_m1, (1, 1, half_dim), device)?;
            let mask_m2 = Tensor::from_vec(mask_m2, (1, 1, half_dim), device)?;
            (mask_m0, mask_m1, mask_m2)
        }
    };

    let cos_m0_masked = cos_m0.broadcast_mul(&mask_m0)?;
    let cos_m1_masked = cos_m1.broadcast_mul(&mask_m1)?;
    let cos_m2_masked = cos_m2.broadcast_mul(&mask_m2)?;
    let cos_half = cos_m0_masked
        .broadcast_add(&cos_m1_masked)?
        .broadcast_add(&cos_m2_masked)?;

    let sin_m0_masked = sin_m0.broadcast_mul(&mask_m0)?;
    let sin_m1_masked = sin_m1.broadcast_mul(&mask_m1)?;
    let sin_m2_masked = sin_m2.broadcast_mul(&mask_m2)?;
    let sin_half = sin_m0_masked
        .broadcast_add(&sin_m1_masked)?
        .broadcast_add(&sin_m2_masked)?;

    let cos_half = cos_half.to_dtype(original_dtype)?.contiguous()?;
    let sin_half = sin_half.to_dtype(original_dtype)?.contiguous()?;

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let q_embed = candle_nn::rotary_emb::rope(&q, &cos_half, &sin_half)?;
    let k_embed = candle_nn::rotary_emb::rope(&k, &cos_half, &sin_half)?;
    Ok((q_embed, k_embed))
}
