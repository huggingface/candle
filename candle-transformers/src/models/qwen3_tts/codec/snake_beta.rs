//! SnakeBeta activation: `x + (1/β) · sin²(α · x)`.

use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

pub struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
    epsilon: f64,
}

impl SnakeBeta {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            alpha: vb.get((channels,), "alpha")?,
            beta: vb.get((channels,), "beta")?,
            epsilon: 1e-9,
        })
    }

    pub fn from_weights(alpha: Tensor, beta: Tensor) -> Result<Self> {
        Ok(Self { alpha, beta, epsilon: 1e-9 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Reshape [C] → [1, C, 1] for broadcasting over [B, C, T]
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(2)?.exp()?;
        let beta = self.beta.unsqueeze(0)?.unsqueeze(2)?.exp()?;
        let sin_sq = x.broadcast_mul(&alpha)?.sin()?.sqr()?;
        let inv_beta = (beta + self.epsilon)?.recip()?;
        Ok((x + sin_sq.broadcast_mul(&inv_beta)?)?)
    }
}

impl Module for SnakeBeta {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        SnakeBeta::forward(self, x)
    }
}
