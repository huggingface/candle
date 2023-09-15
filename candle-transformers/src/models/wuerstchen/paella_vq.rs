use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug)]
pub struct MixingResidualBlock {
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

impl Module for MixingResidualBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mods = &self.gammas;
        let x_temp = xs
            .permute((0, 2, 3, 1))?
            .apply(&self.norm1)?
            .permute((0, 3, 1, 2))?
            .affine(1. + mods[0] as f64, mods[1] as f64)?;
        let x_temp = candle_nn::ops::replication_pad2d(&x_temp, 1)?;
        let xs = (xs + x_temp.apply(&self.depthwise_conv)? * mods[2] as f64)?;
        let x_temp = xs
            .permute((0, 2, 3, 1))?
            .apply(&self.norm2)?
            .permute((0, 3, 1, 2))?
            .affine(1. + mods[3] as f64, mods[4] as f64)?;
        let x_temp = x_temp
            .permute((0, 2, 3, 1))?
            .apply(&self.channelwise_lin1)?
            .gelu()?
            .apply(&self.channelwise_lin2)?
            .permute((0, 3, 1, 2))?;
        xs + x_temp * mods[5] as f64
    }
}

#[derive(Debug)]
struct PaellaVQ {
    in_block_conv: candle_nn::Conv2d,
    out_block_conv: candle_nn::Conv2d,
    down_blocks: Vec<(Option<candle_nn::Conv2d>, MixingResidualBlock)>,
    down_blocks_conv: candle_nn::Conv2d,
    down_blocks_bn: candle_nn::BatchNorm,
    up_blocks_conv: candle_nn::Conv2d,
    up_blocks: Vec<(MixingResidualBlock, Option<candle_nn::ConvTranspose2d>)>,
}

impl PaellaVQ {
    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = candle_nn::ops::pixel_unshuffle(xs, 2)?.apply(&self.in_block_conv)?;
        for down_block in self.down_blocks.iter() {
            if let Some(conv) = &down_block.0 {
                xs = xs.apply(conv)?
            }
            xs = xs.apply(&down_block.1)?
        }
        xs.apply(&self.down_blocks_conv)?
            .apply(&self.down_blocks_bn)
    }

    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        // TODO: quantizer if we want to support `force_not_quantize=False`.
        let mut xs = xs.apply(&self.up_blocks_conv)?;
        for up_block in self.up_blocks.iter() {
            xs = xs.apply(&up_block.0)?;
            if let Some(conv) = &up_block.1 {
                xs = xs.apply(conv)?
            }
        }
        xs.apply(&self.out_block_conv)?
            .apply(&|xs: &_| candle_nn::ops::pixel_shuffle(xs, 2))
    }
}

impl Module for PaellaVQ {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.decode(&self.encode(xs)?)
    }
}
