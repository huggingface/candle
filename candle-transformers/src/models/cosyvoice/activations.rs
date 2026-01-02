//! CosyVoice3 Activation Functions
//!
//! Implements specialized activation functions used in HiFT-GAN vocoder.

use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// Snake activation function: x + (1/α) * sin²(αx)
///
/// This activation is used in HiFi-GAN style vocoders for better
/// audio synthesis quality.
///
/// Reference: Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders
#[derive(Debug, Clone)]
pub struct Snake {
    alpha: Tensor,
    eps: f64,
}

impl Snake {
    /// Create a new Snake activation with learnable alpha parameter.
    ///
    /// # Arguments
    /// * `channels` - Number of channels (alpha is per-channel)
    /// * `vb` - VarBuilder for loading weights
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((channels,), "alpha")?;
        Ok(Self { alpha, eps: 1e-9 })
    }

    /// Create Snake with initialized alpha values (for testing or default init).
    pub fn with_alpha(alpha: Tensor) -> Self {
        Self { alpha, eps: 1e-9 }
    }
}

impl Module for Snake {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T]
        // alpha: [C] -> [1, C, 1]
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(2)?;

        // sin²(αx)
        let ax = x.broadcast_mul(&alpha)?;
        let sin_ax = ax.sin()?;
        let sin_sq = (&sin_ax * &sin_ax)?;

        // x + (1/α) * sin²(αx)
        let inv_alpha = (1.0 / (&alpha + self.eps)?)?;
        let term = sin_sq.broadcast_mul(&inv_alpha)?;
        x + term
    }
}

/// SnakeBeta activation: x + (1/β) * sin²(αx)
///
/// Variant where alpha and beta can be different (used in BigVGAN).
#[derive(Debug, Clone)]
pub struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
    eps: f64,
}

impl SnakeBeta {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((channels,), "alpha")?;
        let beta = vb.get((channels,), "beta")?;
        Ok(Self {
            alpha,
            beta,
            eps: 1e-9,
        })
    }
}

impl Module for SnakeBeta {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(2)?;
        let beta = self.beta.unsqueeze(0)?.unsqueeze(2)?;

        let ax = x.broadcast_mul(&alpha)?;
        let sin_ax = ax.sin()?;
        let sin_sq = (&sin_ax * &sin_ax)?;

        let inv_beta = (1.0 / (&beta + self.eps)?)?;
        let term = sin_sq.broadcast_mul(&inv_beta)?;
        x + term
    }
}

/// ELU activation (used in F0 predictor)
#[derive(Debug, Clone, Copy, Default)]
pub struct Elu {
    alpha: f64,
}

impl Elu {
    pub fn new() -> Self {
        Self { alpha: 1.0 }
    }

    pub fn with_alpha(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Module for Elu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.elu(self.alpha)
    }
}

/// Swish/SiLU activation: x * sigmoid(x)
#[derive(Debug, Clone, Copy, Default)]
pub struct Swish;

impl Module for Swish {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.silu()
    }
}

/// LeakyReLU activation
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    negative_slope: f64,
}

impl LeakyReLU {
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self {
            negative_slope: 0.01,
        }
    }
}

impl Module for LeakyReLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let zeros = x.zeros_like()?;
        let max_val = x.maximum(&zeros)?;
        let min_val = x.minimum(&zeros)?;
        max_val + (min_val * self.negative_slope)?
    }
}

/// Tanh activation wrapper
#[derive(Debug, Clone, Copy, Default)]
pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.tanh()
    }
}

/// Mish activation: x * tanh(softplus(x))
/// Used in CausalConvPositionEmbedding
#[derive(Debug, Clone, Copy, Default)]
pub struct Mish;

impl Module for Mish {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // mish(x) = x * tanh(softplus(x))
        // softplus(x) = ln(1 + exp(x))
        let softplus = (x.exp()? + 1.0)?.log()?;
        let tanh_sp = softplus.tanh()?;
        x * tanh_sp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    fn test_snake_activation() -> Result<()> {
        let device = Device::Cpu;
        let channels = 4;

        // Create alpha tensor
        let alpha = Tensor::ones((channels,), DType::F32, &device)?;
        let snake = Snake::with_alpha(alpha);

        // Test input [batch=1, channels=4, time=3]
        let x = Tensor::randn(0f32, 1.0, (1, channels, 3), &device)?;
        let output = snake.forward(&x)?;

        assert_eq!(output.dims(), &[1, channels, 3]);
        Ok(())
    }

    #[test]
    fn test_leaky_relu() -> Result<()> {
        let device = Device::Cpu;
        let lrelu = LeakyReLU::new(0.1);

        let x = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], 5, &device)?;
        let output = lrelu.forward(&x)?;
        let output_vec: Vec<f32> = output.to_vec1()?;

        // Check negative values are scaled
        assert!((output_vec[0] - (-0.2)).abs() < 1e-5);
        assert!((output_vec[1] - (-0.1)).abs() < 1e-5);
        // Check positive values unchanged
        assert!((output_vec[3] - 1.0).abs() < 1e-5);
        assert!((output_vec[4] - 2.0).abs() < 1e-5);
        Ok(())
    }
}

