use candle::{Result, Tensor, D};
use candle_nn as nn;
use candle_nn::Module;

#[derive(Debug)]
pub struct TimestepEmbedding {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl TimestepEmbedding {
    // act_fn: "silu"
    pub fn new(vs: nn::VarBuilder, channel: usize, time_embed_dim: usize) -> Result<Self> {
        let linear_1 = nn::linear(channel, time_embed_dim, vs.pp("linear_1"))?;
        let linear_2 = nn::linear(time_embed_dim, time_embed_dim, vs.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }
}

impl Module for TimestepEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = nn::ops::silu(&self.linear_1.forward(xs)?)?;
        self.linear_2.forward(&xs)
    }
}

#[derive(Debug)]
pub struct Timesteps {
    num_channels: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
}

impl Timesteps {
    pub fn new(num_channels: usize, flip_sin_to_cos: bool, downscale_freq_shift: f64) -> Self {
        Self {
            num_channels,
            flip_sin_to_cos,
            downscale_freq_shift,
        }
    }
}

impl Module for Timesteps {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let half_dim = (self.num_channels / 2) as u32;
        let exponent = (Tensor::arange(0, half_dim, xs.device())?.to_dtype(candle::DType::F32)?
            * -f64::ln(10000.))?;
        let exponent = (exponent / (half_dim as f64 - self.downscale_freq_shift))?;
        let emb = exponent.exp()?.to_dtype(xs.dtype())?;
        // emb = timesteps[:, None].float() * emb[None, :]
        let emb = xs.unsqueeze(D::Minus1)?.broadcast_mul(&emb.unsqueeze(0)?)?;
        let (cos, sin) = (emb.cos()?, emb.sin()?);
        let emb = if self.flip_sin_to_cos {
            Tensor::cat(&[&cos, &sin], D::Minus1)?
        } else {
            Tensor::cat(&[&sin, &cos], D::Minus1)?
        };
        if self.num_channels % 2 == 1 {
            emb.pad_with_zeros(D::Minus2, 0, 1)
        } else {
            Ok(emb)
        }
    }
}
