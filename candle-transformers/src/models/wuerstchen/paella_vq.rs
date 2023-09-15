#![allow(unused)]
use super::common::{AttnBlock, ResBlock, TimestepBlock};
use candle::{DType, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

#[derive(Debug)]
struct MixingResidualBlock {
    norm1: candle_nn::LayerNorm,
    depthwise_conv: candle_nn::Conv2d,
    norm2: candle_nn::LayerNorm,
    channelwise_lin1: candle_nn::Linear,
    channelwise_lin2: candle_nn::Linear,
    gammas: Vec<f32>,
}

impl MixingResidualBlock {
    pub fn new(inp: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = candle_nn::LayerNormConfig {
            affine: false,
            eps: 1e-6,
            remove_mean: true,
        };
        let norm1 = candle_nn::layer_norm(inp, cfg, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(inp, cfg, vb.pp("norm1"))?;
        let cfg = candle_nn::Conv2dConfig {
            groups: inp,
            ..Default::default()
        };
        let depthwise_conv = candle_nn::conv2d(inp, inp, 3, cfg, vb.pp("depthwise.1"))?;
        let channelwise_lin1 = candle_nn::linear(inp, embed_dim, vb.pp("channelwise.0"))?;
        let channelwise_lin2 = candle_nn::linear(embed_dim, inp, vb.pp("channelwise.2"))?;
        let gammas = vb.get(6, "gammas")?.to_vec1::<f32>()?;
        Ok(Self {
            norm1,
            depthwise_conv,
            norm2,
            channelwise_lin1,
            channelwise_lin2,
            gammas,
        })
    }
}
