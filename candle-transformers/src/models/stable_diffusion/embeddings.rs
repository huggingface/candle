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

impl Timesteps {
    fn freqs_cached(&self, device: &candle::Device, dtype: candle::DType) -> Result<Tensor> {
        use std::sync::{Mutex, OnceLock};
        use std::collections::HashMap;
        // `emb` (the post-exp frequency tensor of shape [1, half_dim]) depends
        // only on (num_channels, downscale_freq_shift, device, dtype).
        // Caching it across denoise steps eliminates a Tensor::arange +
        // to_dtype + mul + div + exp + unsqueeze chain on every step.
        type Key = (usize, u64, candle::DeviceLocation, candle::DType);
        static CACHE: OnceLock<Mutex<HashMap<Key, Tensor>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let key = (
            self.num_channels,
            self.downscale_freq_shift.to_bits(),
            device.location(),
            dtype,
        );
        if let Ok(g) = cache.lock() {
            if let Some(t) = g.get(&key) {
                return Ok(t.clone());
            }
        }
        let half_dim = (self.num_channels / 2) as u32;
        let exponent = (Tensor::arange(0, half_dim, device)?.to_dtype(candle::DType::F32)?
            * -f64::ln(10000.))?;
        let exponent = (exponent / (half_dim as f64 - self.downscale_freq_shift))?;
        let emb = exponent.exp()?.to_dtype(dtype)?.unsqueeze(0)?;
        if let Ok(mut g) = cache.lock() {
            g.insert(key, emb.clone());
        }
        Ok(emb)
    }
}

impl Module for Timesteps {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let emb = self.freqs_cached(xs.device(), xs.dtype())?;
        // emb cached as [1, half_dim]; emit timesteps[:, None] * emb[None, :]
        let emb = xs.unsqueeze(D::Minus1)?.broadcast_mul(&emb)?;
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
