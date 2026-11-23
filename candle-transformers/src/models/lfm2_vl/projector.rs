use crate::models::with_tracing::{layer_norm, linear_b, LayerNorm, Linear};

use candle::{Module, Result, Tensor};

use candle_nn::{Activation, VarBuilder};

use super::config::Lfm2VlConfig;

pub struct Lfm2MultiModalProjector {
    linear1: Linear,
    layer_norm1: Option<LayerNorm>,
    act: Activation,
    linear2: Linear,
    factor: usize,
}

impl Lfm2MultiModalProjector {
    pub fn new(cfg: &Lfm2VlConfig, vb: VarBuilder) -> Result<Self> {
        let in_channels = cfg.vision_config.hidden_size * (cfg.downsample_factor.pow(2));
        let layer_norm1 = if cfg.projector_use_layernorm {
            Some(layer_norm(in_channels, 1e-5, vb.pp("layer_norm"))?)
        } else {
            None
        };
        let linear1 = linear_b(
            in_channels,
            cfg.projector_hidden_size,
            cfg.projector_bias,
            vb.pp("linear_1"),
        )?;
        let act = cfg.projector_hidden_act;
        let linear2 = linear_b(
            cfg.projector_hidden_size,
            cfg.text_config.hidden_size,
            cfg.projector_bias,
            vb.pp("linear_2"),
        )?;

        Ok(Self {
            linear1,
            layer_norm1,
            act,
            linear2,
            factor: cfg.downsample_factor,
        })
    }

    fn pixel_unshuffle(&self, hidden_state: &Tensor) -> Result<Tensor> {
        let (batch_size, width, height, channels) = hidden_state.dims4()?;
        let f = self.factor;

        hidden_state
            .reshape((batch_size, width, height / f, channels * f))?
            .permute((0, 2, 1, 3))?
            .reshape((batch_size, height / f, width / f, channels * f.pow(2)))?
            .permute((0, 2, 1, 3))?
            .contiguous()
    }
}

impl Module for Lfm2MultiModalProjector {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.pixel_unshuffle(xs)?
            .apply(&self.layer_norm1.as_ref())?
            .apply(&self.linear1)?
            .apply(&self.act)?
            .apply(&self.linear2)
    }
}
