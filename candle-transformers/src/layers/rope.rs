use std::{ops::Mul, str::FromStr};

use candle::{DType, Device, IndexOp, Result, Tensor};
use serde::Deserialize;

/// RoPE supporting LongRope
#[derive(Debug, Clone)]
pub struct PhiRotaryEmbedding {
    short_sin: Tensor,
    short_cos: Tensor,
    long_cos: Option<Tensor>,
    long_sin: Option<Tensor>,
    original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub enum ScaledRopeType {
    #[serde(alias = "su")]
    #[serde(alias = "longrope")]
    Su,
    #[serde(alias = "yarn")]
    Yarn,
}

impl FromStr for ScaledRopeType {
    type Err = candle::Error;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "su" | "longrope" => Ok(Self::Su),
            "yarn" => Ok(Self::Yarn),
            _ => Err(candle::Error::Msg(
                "Expected either `su` or `yarn` scaled RoPE type.".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PhiRopeScalingConfig {
    Classic {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
    },
    Scaled {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
        long_mscale: f64,
        short_mscale: f64,
    },
}

pub struct PhiRopeConfig {
    pub rope_scaling: Option<PhiRopeScalingConfig>,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: usize,
    pub rope_theta: f64,
    pub head_dim: usize,
}

impl PhiRotaryEmbedding {
    fn new_classic_scaled(
        short_factor: &[f64],
        long_factor: &[f64],
        scaling_type: &ScaledRopeType,
        cfg: &PhiRopeConfig,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.head_dim;

        // Calculate scale
        let scale =
            cfg.max_position_embeddings as f64 / cfg.original_max_position_embeddings as f64;
        let scaling_factor = if scale <= 1.0 {
            1.0
        } else {
            match scaling_type {
                ScaledRopeType::Su => {
                    (1.0 + scale.ln() / (cfg.original_max_position_embeddings as f64).ln()).sqrt()
                }
                ScaledRopeType::Yarn => 0.1 * scale.ln() + 1.0,
            }
        };

        // Calculate inv freqs for short, long
        let inv_freq_long = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                (1f64 / (long_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64))) as f32
            })
            .collect::<Vec<_>>();
        let inv_freq_short = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                (1f64 / (short_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64))) as f32
            })
            .collect::<Vec<_>>();
        let inv_freq_len = inv_freq_long.len();

        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;

        // Calculate sin,cos for long
        let inv_freq_long = Tensor::from_vec(inv_freq_long, (1, inv_freq_len), dev)?;
        let freqs_long = t.matmul(&inv_freq_long)?;
        let long_sin = freqs_long.sin()?.mul(scaling_factor)?.to_dtype(dtype)?;
        let long_cos = freqs_long.cos()?.mul(scaling_factor)?.to_dtype(dtype)?;

        // Calculate sin,cos for short
        let inv_freq_short =
            Tensor::from_vec(inv_freq_short, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let freqs_short = t.matmul(&inv_freq_short)?;
        let short_sin = freqs_short.sin()?.mul(scaling_factor)?.to_dtype(dtype)?;
        let short_cos = freqs_short.cos()?.mul(scaling_factor)?.to_dtype(dtype)?;

        Ok(Self {
            short_cos,
            short_sin,
            long_cos: Some(long_cos),
            long_sin: Some(long_sin),
            original_max_position_embeddings: cfg.original_max_position_embeddings,
        })
    }

    fn new_unscaled(cfg: &PhiRopeConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.head_dim;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        Ok(Self {
            short_cos: cos,
            short_sin: sin,
            long_cos: None,
            long_sin: None,
            original_max_position_embeddings: cfg.original_max_position_embeddings,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_scaled(
        short_factor: &[f64],
        long_factor: &[f64],
        scaling_type: &ScaledRopeType,
        long_mscale: f64,
        short_mscale: f64,
        cfg: &PhiRopeConfig,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.head_dim;

        if !matches!(scaling_type, ScaledRopeType::Su) {
            candle::bail!("Scaled Phi3 RoPE (non-classic scaled, with mscales) must have type `su`/`longrope`.");
        }

        if short_factor.len() != dim / 2 {
            candle::bail!(
                "Misaligned length {}, expected {} for `su`/`longrope` short rescale factors",
                short_factor.len(),
                dim / 2
            );
        }
        if long_factor.len() != dim / 2 {
            candle::bail!(
                "Misaligned length {}, expected {} for `su`/`longrope` long rescale factors",
                long_factor.len(),
                dim / 2
            );
        }

        // Short cos/sin
        let inv_freq_short: Vec<_> = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                1f32 / (short_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64)) as f32
            })
            .collect();
        let inv_freq_len_short = inv_freq_short.len();
        let inv_freq_short = Tensor::from_vec(inv_freq_short, (1, inv_freq_len_short), dev)?;
        let t_short = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs_short = t_short.matmul(&inv_freq_short)?;
        let sin_short = (freqs_short.sin()?.to_dtype(dtype)? * short_mscale)?;
        let cos_short = (freqs_short.cos()?.to_dtype(dtype)? * short_mscale)?;

        // Long cos/sin
        let inv_freq_long: Vec<_> = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                1f32 / (long_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64)) as f32
            })
            .collect();
        let inv_freq_len_long = inv_freq_long.len();
        let inv_freq_long = Tensor::from_vec(inv_freq_long, (1, inv_freq_len_long), dev)?;
        let t_long = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs_long = t_long.matmul(&inv_freq_long)?;
        let sin_long = (freqs_long.sin()?.to_dtype(dtype)? * long_mscale)?;
        let cos_long = (freqs_long.cos()?.to_dtype(dtype)? * long_mscale)?;
        Ok(Self {
            short_cos: cos_short,
            short_sin: sin_short,
            long_cos: Some(cos_long),
            long_sin: Some(sin_long),
            original_max_position_embeddings: cfg.original_max_position_embeddings,
        })
    }

    pub fn new(dtype: DType, cfg: impl Into<PhiRopeConfig>, dev: &Device) -> Result<Self> {
        let cfg: PhiRopeConfig = cfg.into();

        match &cfg.rope_scaling {
            Some(PhiRopeScalingConfig::Classic {
                short_factor,
                long_factor,
                scaling_type,
            }) => {
                Self::new_classic_scaled(short_factor, long_factor, scaling_type, &cfg, dtype, dev)
            }

            Some(PhiRopeScalingConfig::Scaled {
                short_factor,
                long_factor,
                scaling_type,
                long_mscale,
                short_mscale,
            }) => Self::new_scaled(
                short_factor,
                long_factor,
                scaling_type,
                *long_mscale,
                *short_mscale,
                &cfg,
                dtype,
                dev,
            ),

            None => Self::new_unscaled(&cfg, dtype, dev),
        }
    }

    /// Returns (sin, cos) taking into account LongRope
    fn get_long_or_short_sin_cos(&self, position_ids: &[usize]) -> (&Tensor, &Tensor) {
        if self.long_cos.is_none() {
            return (&self.short_sin, &self.short_cos);
        }
        let seq_len = position_ids.iter().max().unwrap() + 1;
        if seq_len > self.original_max_position_embeddings {
            (
                self.long_sin.as_ref().unwrap(),
                self.long_cos.as_ref().unwrap(),
            )
        } else {
            (&self.short_sin, &self.short_cos)
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let mut q_embeds = Vec::new();
        let mut k_embeds = Vec::new();
        let (sin, cos) = self.get_long_or_short_sin_cos(position_ids);
        for (i, offset) in seqlen_offsets.iter().enumerate() {
            let cos = cos.narrow(0, *offset, seq_len)?;
            let sin = sin.narrow(0, *offset, seq_len)?;
            let q_embed =
                candle_nn::rotary_emb::rope(&q.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
            let k_embed =
                candle_nn::rotary_emb::rope(&k.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
            q_embeds.push(q_embed);
            k_embeds.push(k_embed);
        }
        Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
    }
}
