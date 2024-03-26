use super::common::LayerNormNoWeights;
use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug)]
pub struct MixingResidualBlock {
    norm1: LayerNormNoWeights,
    depthwise_conv: candle_nn::Conv2d,
    norm2: LayerNormNoWeights,
    channelwise_lin1: candle_nn::Linear,
    channelwise_lin2: candle_nn::Linear,
    gammas: Vec<f32>,
}

impl MixingResidualBlock {
    pub fn new(inp: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = LayerNormNoWeights::new(inp)?;
        let norm2 = LayerNormNoWeights::new(inp)?;
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
            .contiguous()?
            .apply(&self.channelwise_lin1)?
            .gelu()?
            .apply(&self.channelwise_lin2)?
            .permute((0, 3, 1, 2))?;
        xs + x_temp * mods[5] as f64
    }
}

#[derive(Debug)]
pub struct PaellaVQ {
    in_block_conv: candle_nn::Conv2d,
    out_block_conv: candle_nn::Conv2d,
    down_blocks: Vec<(Option<candle_nn::Conv2d>, MixingResidualBlock)>,
    down_blocks_conv: candle_nn::Conv2d,
    down_blocks_bn: candle_nn::BatchNorm,
    up_blocks_conv: candle_nn::Conv2d,
    up_blocks: Vec<(Vec<MixingResidualBlock>, Option<candle_nn::ConvTranspose2d>)>,
}

impl PaellaVQ {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        const IN_CHANNELS: usize = 3;
        const OUT_CHANNELS: usize = 3;
        const LATENT_CHANNELS: usize = 4;
        const EMBED_DIM: usize = 384;
        const BOTTLENECK_BLOCKS: usize = 12;
        const C_LEVELS: [usize; 2] = [EMBED_DIM / 2, EMBED_DIM];

        let in_block_conv = candle_nn::conv2d(
            IN_CHANNELS * 4,
            C_LEVELS[0],
            1,
            Default::default(),
            vb.pp("in_block.1"),
        )?;
        let out_block_conv = candle_nn::conv2d(
            C_LEVELS[0],
            OUT_CHANNELS * 4,
            1,
            Default::default(),
            vb.pp("out_block.0"),
        )?;

        let mut down_blocks = Vec::new();
        let vb_d = vb.pp("down_blocks");
        let mut d_idx = 0;
        for (i, &c_level) in C_LEVELS.iter().enumerate() {
            let conv_block = if i > 0 {
                let cfg = candle_nn::Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                };
                let block = candle_nn::conv2d(C_LEVELS[i - 1], c_level, 4, cfg, vb_d.pp(d_idx))?;
                d_idx += 1;
                Some(block)
            } else {
                None
            };
            let res_block = MixingResidualBlock::new(c_level, c_level * 4, vb_d.pp(d_idx))?;
            d_idx += 1;
            down_blocks.push((conv_block, res_block))
        }
        let vb_d = vb_d.pp(d_idx);
        let down_blocks_conv = candle_nn::conv2d_no_bias(
            C_LEVELS[1],
            LATENT_CHANNELS,
            1,
            Default::default(),
            vb_d.pp(0),
        )?;
        let down_blocks_bn = candle_nn::batch_norm(LATENT_CHANNELS, 1e-5, vb_d.pp(1))?;

        let mut up_blocks = Vec::new();
        let vb_u = vb.pp("up_blocks");
        let mut u_idx = 0;
        let up_blocks_conv = candle_nn::conv2d(
            LATENT_CHANNELS,
            C_LEVELS[1],
            1,
            Default::default(),
            vb_u.pp(u_idx).pp(0),
        )?;
        u_idx += 1;
        for (i, &c_level) in C_LEVELS.iter().rev().enumerate() {
            let mut res_blocks = Vec::new();
            let n_bottleneck_blocks = if i == 0 { BOTTLENECK_BLOCKS } else { 1 };
            for _j in 0..n_bottleneck_blocks {
                let res_block = MixingResidualBlock::new(c_level, c_level * 4, vb_u.pp(u_idx))?;
                u_idx += 1;
                res_blocks.push(res_block)
            }
            let conv_block = if i < C_LEVELS.len() - 1 {
                let cfg = candle_nn::ConvTranspose2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                };
                let block = candle_nn::conv_transpose2d(
                    c_level,
                    C_LEVELS[C_LEVELS.len() - i - 2],
                    4,
                    cfg,
                    vb_u.pp(u_idx),
                )?;
                u_idx += 1;
                Some(block)
            } else {
                None
            };
            up_blocks.push((res_blocks, conv_block))
        }
        Ok(Self {
            in_block_conv,
            down_blocks,
            down_blocks_conv,
            down_blocks_bn,
            up_blocks,
            up_blocks_conv,
            out_block_conv,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = candle_nn::ops::pixel_unshuffle(xs, 2)?.apply(&self.in_block_conv)?;
        for down_block in self.down_blocks.iter() {
            if let Some(conv) = &down_block.0 {
                xs = xs.apply(conv)?
            }
            xs = xs.apply(&down_block.1)?
        }
        xs.apply(&self.down_blocks_conv)?
            .apply_t(&self.down_blocks_bn, false)
    }

    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        // TODO: quantizer if we want to support `force_not_quantize=False`.
        let mut xs = xs.apply(&self.up_blocks_conv)?;
        for up_block in self.up_blocks.iter() {
            for b in up_block.0.iter() {
                xs = xs.apply(b)?;
            }
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
