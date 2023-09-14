#![allow(unused)]
use super::common::{GlobalResponseNorm, ResBlock, TimestepBlock, WLayerNorm};
use candle::{DType, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

#[derive(Debug)]
pub struct ResBlockStageB {
    depthwise: candle_nn::Conv2d,
    norm: WLayerNorm,
    channelwise_lin1: candle_nn::Linear,
    channelwise_grn: GlobalResponseNorm,
    channelwise_lin2: candle_nn::Linear,
}

impl ResBlockStageB {
    pub fn new(c: usize, c_skip: usize, ksize: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = candle_nn::Conv2dConfig {
            groups: c,
            padding: ksize / 2,
            ..Default::default()
        };
        let depthwise = candle_nn::conv2d(c, c, ksize, cfg, vb.pp("depthwise"))?;
        let norm = WLayerNorm::new(c, vb.pp("norm"))?;
        let channelwise_lin1 = candle_nn::linear(c + c_skip, c * 4, vb.pp("channelwise.0"))?;
        let channelwise_grn = GlobalResponseNorm::new(4 * c, vb.pp("channelwise.2"))?;
        let channelwise_lin2 = candle_nn::linear(c * 4, c, vb.pp("channelwise.4"))?;
        Ok(Self {
            depthwise,
            norm,
            channelwise_lin1,
            channelwise_grn,
            channelwise_lin2,
        })
    }

    pub fn forward(&self, xs: &Tensor, x_skip: Option<&Tensor>) -> Result<Tensor> {
        let x_res = xs;
        let xs = xs.apply(&self.depthwise)?.apply(&self.norm)?;
        let xs = match x_skip {
            None => xs.clone(),
            Some(x_skip) => Tensor::cat(&[&xs, x_skip], 1)?,
        };
        let xs = xs
            .permute((0, 2, 3, 1))?
            .apply(&self.channelwise_lin1)?
            .gelu()?
            .apply(&self.channelwise_grn)?
            .apply(&self.channelwise_lin2)?
            .permute((0, 3, 1, 2))?;
        xs + x_res
    }
}

#[derive(Debug)]
pub struct WDiffNeXt {}
