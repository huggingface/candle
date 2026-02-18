//! RoPE scaling variants (linear, dynamic, yarn, longrope, llama3).
//!
//! Ported from `qwen3-tts-rs` with small adaptations.

use candle::{Device, Result, Tensor};

/// RoPE scaling type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopeScalingType {
    /// No scaling (standard RoPE).
    #[default]
    Default,
    /// Linear interpolation for longer sequences.
    Linear,
    /// NTK-aware dynamic scaling.
    Dynamic,
    /// YaRN (Yet Another RoPE extensioN).
    Yarn,
    /// Long context RoPE with short/long factors.
    LongRope,
    /// Llama 3's frequency-based scaling.
    Llama3,
}

impl RopeScalingType {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "linear" => Self::Linear,
            "dynamic" | "ntk" => Self::Dynamic,
            "yarn" => Self::Yarn,
            "longrope" | "long_rope" => Self::LongRope,
            "llama3" | "llama_3" => Self::Llama3,
            _ => Self::Default,
        }
    }
}

pub fn compute_linear_scaling(inv_freq: &Tensor, factor: f64) -> Result<Tensor> {
    inv_freq.affine(1.0 / factor, 0.0)
}

pub fn compute_dynamic_scaling(
    head_dim: usize,
    rope_theta: f64,
    seq_len: usize,
    original_max_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let base = if seq_len > original_max_len {
        let ratio = seq_len as f64 / original_max_len as f64;
        rope_theta
            * (ratio * (head_dim as f64 / (head_dim as f64 - 2.0)))
                .powf(head_dim as f64 / (head_dim as f64 - 2.0))
    } else {
        rope_theta
    };

    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (base.powf(i as f64 * 2.0 / head_dim as f64) as f32))
        .collect();
    Tensor::from_vec(inv_freq, half_dim, device)
}

pub fn compute_yarn_scaling(
    head_dim: usize,
    rope_theta: f64,
    factor: f64,
    original_max_len: usize,
    beta_fast: f64,
    beta_slow: f64,
    device: &Device,
) -> Result<(Tensor, f64)> {
    let half_dim = head_dim / 2;

    let low_freq_factor = 1.0 / (beta_fast / (beta_fast - beta_slow));
    let high_freq_factor = 1.0 / (1.0 - (beta_slow / (beta_fast - beta_slow)));

    let low_freq_wavelen = original_max_len as f64 / low_freq_factor;
    let high_freq_wavelen = original_max_len as f64 / high_freq_factor;

    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| {
            let freq = 1.0 / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64));
            let wavelen = 2.0 * std::f64::consts::PI / freq;

            let scaled_freq = if wavelen < high_freq_wavelen {
                freq
            } else if wavelen > low_freq_wavelen {
                freq / factor
            } else {
                let smooth = (wavelen - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen);
                freq * (1.0 - smooth) + (freq / factor) * smooth
            };
            scaled_freq as f32
        })
        .collect();

    let attention_factor = (1.0 + (factor.ln() / (original_max_len as f64).ln())).sqrt();

    Ok((
        Tensor::from_vec(inv_freq, half_dim, device)?,
        attention_factor,
    ))
}

pub fn compute_longrope_scaling(
    head_dim: usize,
    rope_theta: f64,
    seq_len: usize,
    original_max_len: usize,
    short_factor: &[f64],
    long_factor: &[f64],
    device: &Device,
) -> Result<Tensor> {
    let half_dim = head_dim / 2;
    let factors = if seq_len > original_max_len {
        long_factor
    } else {
        short_factor
    };

    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| {
            let base_freq = 1.0 / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64));
            let factor = factors.get(i).copied().unwrap_or(1.0);
            (base_freq / factor) as f32
        })
        .collect();

    Tensor::from_vec(inv_freq, half_dim, device)
}

pub fn compute_llama3_scaling(
    head_dim: usize,
    rope_theta: f64,
    factor: f64,
    original_max_len: usize,
    low_freq_factor: f64,
    high_freq_factor: f64,
    device: &Device,
) -> Result<Tensor> {
    let half_dim = head_dim / 2;

    let low_freq_wavelen = original_max_len as f64 / low_freq_factor;
    let high_freq_wavelen = original_max_len as f64 / high_freq_factor;

    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| {
            let freq = 1.0 / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64));
            let wavelen = 2.0 * std::f64::consts::PI / freq;

            let scaled_freq = if wavelen < high_freq_wavelen {
                freq
            } else if wavelen > low_freq_wavelen {
                freq / factor
            } else {
                let smooth = (original_max_len as f64 / wavelen - low_freq_factor)
                    / (high_freq_factor - low_freq_factor);
                freq * (1.0 - smooth) + (freq / factor) * smooth
            };
            scaled_freq as f32
        })
        .collect();

    Tensor::from_vec(inv_freq, half_dim, device)
}
