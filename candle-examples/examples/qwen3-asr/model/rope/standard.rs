//! Standard RoPE embedding generator.

use candle::{DType, Device, Result, Tensor};

use crate::config::RopeScaling;
use crate::model::rope::core::RopeCore;
use crate::model::rope::scaling::RopeScalingType;

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    core: RopeCore,
}

impl RotaryEmbedding {
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

    /// Compute cos and sin for the given `position_ids`.
    ///
    /// Returns half-dim embeddings (head_dim/2) suitable for `candle_nn::rotary_emb::rope`.
    pub fn forward(&self, x: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let dtype = x.dtype();
        let seq_len = position_ids.dim(1)?;

        let inv_freq = self.core.get_inv_freq(seq_len)?;

        // inv_freq: (half_dim,) -> (1, half_dim, 1)
        let inv_freq = inv_freq.unsqueeze(0)?.unsqueeze(2)?.to_dtype(DType::F32)?;

        // position_ids: (batch, seq_len) -> (batch, 1, seq_len)
        let position_ids = position_ids.unsqueeze(1)?.to_dtype(DType::F32)?;

        // freqs: (batch, half_dim, seq_len) -> (batch, seq_len, half_dim)
        let freqs = inv_freq.broadcast_mul(&position_ids)?;
        let freqs = freqs.transpose(1, 2)?.contiguous()?;

        let cos = (freqs.cos()? * self.core.attention_scaling)?.to_dtype(dtype)?;
        let sin = (freqs.sin()? * self.core.attention_scaling)?.to_dtype(dtype)?;

        Ok((cos, sin))
    }
}
