//! Shared RoPE core computation and state.

use candle::{Device, Result, Tensor};

use crate::config::RopeScaling;
use crate::model::rope::scaling::{
    compute_dynamic_scaling, compute_linear_scaling, compute_llama3_scaling,
    compute_longrope_scaling, compute_yarn_scaling, RopeScalingType,
};

/// Shared RoPE computation core.
#[derive(Debug, Clone)]
pub struct RopeCore {
    pub inv_freq: Tensor,
    pub attention_scaling: f64,
    pub original_max_len: usize,
    pub scaling_type: RopeScalingType,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub device: Device,
    pub short_factor: Vec<f64>,
    pub long_factor: Vec<f64>,
}

impl RopeCore {
    pub fn new(
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq = Self::compute_default_inv_freq(head_dim, rope_theta, device)?;
        Ok(Self {
            inv_freq,
            attention_scaling: 1.0,
            original_max_len: max_position_embeddings,
            scaling_type: RopeScalingType::Default,
            head_dim,
            rope_theta,
            device: device.clone(),
            short_factor: Vec::new(),
            long_factor: Vec::new(),
        })
    }

    pub fn with_scaling(
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        scaling: &RopeScaling,
        device: &Device,
    ) -> Result<Self> {
        let scaling_type = scaling
            .rope_type
            .as_deref()
            .map(RopeScalingType::parse)
            .unwrap_or(RopeScalingType::Default);

        let original_max_len = scaling
            .original_max_position_embeddings
            .unwrap_or(max_position_embeddings);

        let scaling_factor = scaling.factor.unwrap_or(1.0);
        let beta_fast = scaling.beta_fast.unwrap_or(32.0);
        let beta_slow = scaling.beta_slow.unwrap_or(1.0);
        let low_freq_factor = scaling.low_freq_factor.unwrap_or(1.0);
        let high_freq_factor = scaling.high_freq_factor.unwrap_or(4.0);
        let short_factor = scaling.short_factor.clone().unwrap_or_default();
        let long_factor = scaling.long_factor.clone().unwrap_or_default();

        let (inv_freq, attention_scaling) = match scaling_type {
            RopeScalingType::Default => (
                Self::compute_default_inv_freq(head_dim, rope_theta, device)?,
                1.0,
            ),
            RopeScalingType::Linear => {
                let base = Self::compute_default_inv_freq(head_dim, rope_theta, device)?;
                (compute_linear_scaling(&base, scaling_factor)?, 1.0)
            }
            RopeScalingType::Dynamic => (
                // Recomputed at forward time.
                Self::compute_default_inv_freq(head_dim, rope_theta, device)?,
                1.0,
            ),
            RopeScalingType::Yarn => compute_yarn_scaling(
                head_dim,
                rope_theta,
                scaling_factor,
                original_max_len,
                beta_fast,
                beta_slow,
                device,
            )?,
            RopeScalingType::LongRope => (
                // Recomputed at forward time.
                Self::compute_default_inv_freq(head_dim, rope_theta, device)?,
                scaling.attention_factor.unwrap_or(1.0),
            ),
            RopeScalingType::Llama3 => (
                compute_llama3_scaling(
                    head_dim,
                    rope_theta,
                    scaling_factor,
                    original_max_len,
                    low_freq_factor,
                    high_freq_factor,
                    device,
                )?,
                1.0,
            ),
        };

        Ok(Self {
            inv_freq,
            attention_scaling,
            original_max_len,
            scaling_type,
            head_dim,
            rope_theta,
            device: device.clone(),
            short_factor,
            long_factor,
        })
    }

    pub fn compute_default_inv_freq(
        head_dim: usize,
        rope_theta: f64,
        device: &Device,
    ) -> Result<Tensor> {
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32))
            .collect();
        Tensor::from_vec(inv_freq, half_dim, device)
    }

    pub fn get_inv_freq(&self, seq_len: usize) -> Result<Tensor> {
        match self.scaling_type {
            RopeScalingType::Dynamic => compute_dynamic_scaling(
                self.head_dim,
                self.rope_theta,
                seq_len,
                self.original_max_len,
                &self.device,
            ),
            RopeScalingType::LongRope
                if !self.short_factor.is_empty() && !self.long_factor.is_empty() =>
            {
                compute_longrope_scaling(
                    self.head_dim,
                    self.rope_theta,
                    seq_len,
                    self.original_max_len,
                    &self.short_factor,
                    &self.long_factor,
                    &self.device,
                )
            }
            _ => Ok(self.inv_freq.clone()),
        }
    }
}
