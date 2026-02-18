//! Multimodal RoPE (mRoPE) helpers.

use candle::{DType, Device, IndexOp, Result, Tensor};

use crate::config::RopeScaling;
use crate::model::rope::core::RopeCore;
use crate::model::rope::scaling::RopeScalingType;

/// Multimodal rotary embedding generator for 3D positions.
///
/// `position_ids` is shaped `(3, batch, seq_len)` where the leading dimension
/// corresponds to `[temporal, height, width]`.
#[derive(Debug, Clone)]
pub struct MultimodalRotaryEmbedding {
    core: RopeCore,
}

impl MultimodalRotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        device: &Device,
    ) -> Result<Self> {
        let core = RopeCore::new(head_dim, max_position_embeddings, rope_theta, device)?;
        Ok(Self { core })
    }

    pub fn with_scaling(
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        scaling: &RopeScaling,
        device: &Device,
    ) -> Result<Self> {
        let core = RopeCore::with_scaling(
            head_dim,
            max_position_embeddings,
            rope_theta,
            scaling,
            device,
        )?;
        Ok(Self { core })
    }

    pub fn attention_scaling(&self) -> f64 {
        self.core.attention_scaling
    }

    pub fn scaling_type(&self) -> RopeScalingType {
        self.core.scaling_type
    }

    /// Compute cos and sin for multimodal positions.
    ///
    /// Returns half-dim embeddings of shape `(3, batch, seq_len, head_dim/2)`.
    pub fn forward(&self, x: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let dtype = x.dtype();
        let seq_len = position_ids.dim(2)?;

        let inv_freq = self.core.get_inv_freq(seq_len)?;

        // inv_freq: (half_dim,) -> (1, 1, half_dim, 1)
        let inv_freq = inv_freq
            .unsqueeze(0)?
            .unsqueeze(0)?
            .unsqueeze(3)?
            .to_dtype(DType::F32)?;

        // position_ids: (3, batch, seq_len) -> (3, batch, 1, seq_len)
        let position_ids = position_ids.unsqueeze(2)?.to_dtype(DType::F32)?;

        // freqs: (3, batch, half_dim, seq_len) -> (3, batch, seq_len, half_dim)
        let freqs = inv_freq.broadcast_mul(&position_ids)?;
        let freqs = freqs.transpose(2, 3)?.contiguous()?;

        let cos = (freqs.cos()? * self.core.attention_scaling)?.to_dtype(dtype)?;
        let sin = (freqs.sin()? * self.core.attention_scaling)?.to_dtype(dtype)?;

        Ok((cos, sin))
    }
}

/// Apply multimodal rotary position embedding for 3D positions.
pub fn apply_multimodal_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
    interleaved: bool,
) -> Result<(Tensor, Tensor)> {
    if interleaved {
        apply_multimodal_rotary_pos_emb_interleaved(q, k, cos, sin, mrope_section)
    } else {
        apply_multimodal_rotary_pos_emb_standard(q, k, cos, sin, mrope_section)
    }
}

fn apply_multimodal_rotary_pos_emb_standard(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor)> {
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

fn apply_multimodal_rotary_pos_emb_interleaved(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor)> {
    let (_modalities, _batch, _seq_len, half_dim) = cos.dims4()?;
    let modality_num = mrope_section.len();

    let original_dtype = cos.dtype();
    let cos_half = cos.contiguous()?.to_dtype(DType::F32)?;
    let sin_half = sin.contiguous()?.to_dtype(DType::F32)?;

    let m1_end = if mrope_section.len() > 1 {
        (mrope_section[1] * modality_num).min(half_dim)
    } else {
        0
    };
    let m2_end = if mrope_section.len() > 2 {
        (mrope_section[2] * modality_num).min(half_dim)
    } else {
        0
    };

    let cos_m0 = cos_half.i(0)?.contiguous()?;
    let sin_m0 = sin_half.i(0)?.contiguous()?;
    let cos_m1 = cos_half.i(1)?.contiguous()?;
    let sin_m1 = sin_half.i(1)?.contiguous()?;
    let cos_m2 = cos_half.i(2)?.contiguous()?;
    let sin_m2 = sin_half.i(2)?.contiguous()?;

    let mut cos_parts: Vec<Tensor> = Vec::with_capacity(half_dim);
    let mut sin_parts: Vec<Tensor> = Vec::with_capacity(half_dim);

    for pos in 0..half_dim {
        let modality = if modality_num >= 3 && mrope_section.len() >= 3 {
            if pos >= 1 && pos < m1_end && (pos - 1) % modality_num == 0 {
                1
            } else if pos >= 2 && pos < m2_end && (pos - 2) % modality_num == 0 {
                2
            } else {
                0
            }
        } else {
            0
        };

        let (cos_src, sin_src) = if modality == 0 {
            (&cos_m0, &sin_m0)
        } else if modality == 1 {
            (&cos_m1, &sin_m1)
        } else if modality == 2 {
            (&cos_m2, &sin_m2)
        } else {
            candle::bail!("invalid modality={modality} for interleaved mRoPE")
        };

        let cos_col = cos_src.narrow(2, pos, 1)?;
        let sin_col = sin_src.narrow(2, pos, 1)?;
        cos_parts.push(cos_col);
        sin_parts.push(sin_col);
    }

    let cos_half = Tensor::cat(&cos_parts.iter().collect::<Vec<_>>(), 2)?;
    let sin_half = Tensor::cat(&sin_parts.iter().collect::<Vec<_>>(), 2)?;

    let cos_half = cos_half.to_dtype(original_dtype)?.contiguous()?;
    let sin_half = sin_half.to_dtype(original_dtype)?.contiguous()?;

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let q_embed = candle_nn::rotary_emb::rope(&q, &cos_half, &sin_half)?;
    let k_embed = candle_nn::rotary_emb::rope(&k, &cos_half, &sin_half)?;
    Ok((q_embed, k_embed))
}

#[cfg(test)]
mod tests {
    use super::{apply_multimodal_rotary_pos_emb, MultimodalRotaryEmbedding};

    #[test]
    fn test_multimodal_rope_shapes() -> anyhow::Result<()> {
        let device = candle::Device::Cpu;

        // head_dim=128 => half_dim=64, mrope_section sums to 64.
        let head_dim = 128usize;
        let rope = MultimodalRotaryEmbedding::new(head_dim, 1024, 10000.0, &device)?;

        let batch = 2usize;
        let seq_len = 7usize;

        // Dummy x: only dtype/device are used by forward.
        let x = candle::Tensor::zeros((batch, 1, seq_len, head_dim), candle::DType::F32, &device)?;

        // position_ids: (3, batch, seq_len)
        let pos1 = candle::Tensor::arange(0i64, seq_len as i64, &device)?.unsqueeze(0)?;
        let pos1 = pos1.broadcast_as((batch, seq_len))?;
        let position_ids = candle::Tensor::stack(&[&pos1, &pos1, &pos1], 0)?;

        let (cos, sin) = rope.forward(&x, &position_ids)?;
        let (m, b, s, half) = cos.dims4()?;
        if (m, b, s, half) != (3, batch, seq_len, head_dim / 2) {
            anyhow::bail!("unexpected cos dims: {:?}", cos.dims());
        }
        let (m, b, s, half) = sin.dims4()?;
        if (m, b, s, half) != (3, batch, seq_len, head_dim / 2) {
            anyhow::bail!("unexpected sin dims: {:?}", sin.dims());
        }

        // Apply mRoPE to q/k (shape checks only).
        let q = candle::Tensor::zeros((batch, 4, seq_len, head_dim), candle::DType::F32, &device)?;
        let k = candle::Tensor::zeros((batch, 4, seq_len, head_dim), candle::DType::F32, &device)?;
        let mrope_section = &[24usize, 20, 20];

        let (q1, k1) = apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin, mrope_section, false)?;
        if q1.dims() != q.dims() || k1.dims() != k.dims() {
            anyhow::bail!("unexpected output dims for standard mRoPE");
        }

        let (q2, k2) = apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin, mrope_section, true)?;
        if q2.dims() != q.dims() || k2.dims() != k.dims() {
            anyhow::bail!("unexpected output dims for interleaved mRoPE");
        }

        Ok(())
    }
}
