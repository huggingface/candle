//! Sampling utilities for Z-Image model.

use candle::{DType, Device, Result, Tensor};

/// Generate initial Gaussian noise
///
/// # Arguments
/// * `batch_size` - Batch size
/// * `channels` - Number of channels (typically 16, VAE latent channels)
/// * `height` - Height (latent space, i.e., image_height / 16)
/// * `width` - Width (latent space)
/// * `device` - Compute device
///
/// # Returns
/// Noise tensor of shape (batch_size, channels, height, width)
pub fn get_noise(
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    device: &Device,
) -> Result<Tensor> {
    Tensor::randn(0f32, 1.0, (batch_size, channels, height, width), device)
}

/// Get linear time schedule with shift
///
/// # Arguments
/// * `num_steps` - Number of inference steps
/// * `mu` - Time shift parameter (from calculate_shift)
///
/// # Returns
/// Time points from 1.0 to 0.0 (num_steps+1 points)
pub fn get_schedule(num_steps: usize, mu: f64) -> Vec<f64> {
    let timesteps: Vec<f64> = (0..=num_steps)
        .map(|v| v as f64 / num_steps as f64)
        .rev()
        .collect();

    // Apply time shift (for Flow Matching)
    timesteps
        .into_iter()
        .map(|t| {
            if t <= 0.0 || t >= 1.0 {
                t // boundary case
            } else {
                let e = mu.exp();
                e / (e + (1.0 / t - 1.0))
            }
        })
        .collect()
}

/// Post-process image from VAE output
/// Converts from [-1, 1] to [0, 255] u8 image
pub fn postprocess_image(image: &Tensor) -> Result<Tensor> {
    let image = image.clamp(-1.0, 1.0)?;
    let image = ((image + 1.0)? * 127.5)?;
    image.to_dtype(DType::U8)
}

/// CFG configuration
#[derive(Debug, Clone)]
pub struct CfgConfig {
    /// Guidance scale (typically 5.0)
    pub guidance_scale: f64,
    /// CFG truncation threshold (1.0 = full CFG, 0.0 = no CFG)
    pub cfg_truncation: f64,
    /// Whether to normalize CFG output
    pub cfg_normalization: bool,
}

impl Default for CfgConfig {
    fn default() -> Self {
        Self {
            guidance_scale: 5.0,
            cfg_truncation: 1.0,
            cfg_normalization: false,
        }
    }
}

/// Apply Classifier-Free Guidance
///
/// # Arguments
/// * `pos_pred` - Positive (conditional) prediction
/// * `neg_pred` - Negative (unconditional) prediction
/// * `cfg` - CFG configuration
/// * `t_norm` - Normalized time [0, 1]
pub fn apply_cfg(
    pos_pred: &Tensor,
    neg_pred: &Tensor,
    cfg: &CfgConfig,
    t_norm: f64,
) -> Result<Tensor> {
    // CFG truncation: disable CFG in late sampling
    let current_scale = if t_norm > cfg.cfg_truncation {
        0.0
    } else {
        cfg.guidance_scale
    };

    if current_scale <= 0.0 {
        return Ok(pos_pred.clone());
    }

    // CFG formula: pred = pos + scale * (pos - neg)
    let diff = (pos_pred - neg_pred)?;
    let pred = (pos_pred + (diff * current_scale)?)?;

    // Optional: CFG normalization (limit output norm)
    if cfg.cfg_normalization {
        let ori_norm = pos_pred.sqr()?.sum_all()?.sqrt()?;
        let new_norm = pred.sqr()?.sum_all()?.sqrt()?;
        let ori_norm_val = ori_norm.to_scalar::<f32>()?;
        let new_norm_val = new_norm.to_scalar::<f32>()?;

        if new_norm_val > ori_norm_val {
            let scale = ori_norm_val / new_norm_val;
            return pred * scale as f64;
        }
    }

    Ok(pred)
}

/// Scale latents to initial noise level
///
/// For flow matching, the initial sample should be pure noise.
/// This function scales the noise by the initial sigma.
pub fn scale_noise(noise: &Tensor, sigma: f64) -> Result<Tensor> {
    noise * sigma
}
