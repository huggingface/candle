#![allow(unused)]
use super::common::{AttnBlock, GlobalResponseNorm, ResBlock, TimestepBlock, WLayerNorm};
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
struct SubBlock {
    res_block: ResBlockStageB,
    ts_block: TimestepBlock,
    attn_block: Option<AttnBlock>,
}

#[derive(Debug)]
struct DownBlock {
    layer_norm: Option<WLayerNorm>,
    conv: Option<candle_nn::Conv2d>,
    sub_blocks: Vec<SubBlock>,
}

#[derive(Debug)]
struct UpBlock {
    sub_blocks: Vec<SubBlock>,
    layer_norm: Option<WLayerNorm>,
    conv: Option<candle_nn::Conv2d>,
}

#[derive(Debug)]
pub struct WDiffNeXt {
    clip_mapper: candle_nn::Linear,
    effnet_mappers: Vec<candle_nn::Conv2d>,
    seq_norm: candle_nn::LayerNorm,
    embedding_conv: candle_nn::Conv2d,
    embedding_ln: WLayerNorm,
    down_blocks: Vec<DownBlock>,
    up_blocks: Vec<UpBlock>,
    clf_ln: WLayerNorm,
    clf_conv: candle_nn::Conv2d,
    c_r: usize,
}

impl WDiffNeXt {
    pub fn new(
        c_in: usize,
        c_out: usize,
        c_r: usize,
        c_cond: usize,
        clip_embd: usize,
        patch_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        const C_HIDDEN: [usize; 4] = [320, 640, 1280, 1280];
        const BLOCKS: [usize; 4] = [4, 4, 14, 4];
        const NHEAD: [usize; 4] = [0, 10, 20, 20];
        const INJECT_EFFNET: [bool; 4] = [false, true, true, true];

        let clip_mapper = candle_nn::linear(clip_embd, c_cond, vb.pp("clip_mapper"))?;
        let effnet_mappers = vec![];
        let cfg = candle_nn::layer_norm::LayerNormConfig {
            ..Default::default()
        };
        let seq_norm = candle_nn::layer_norm(c_cond, cfg, vb.pp("seq_norm"))?;
        let embedding_ln = WLayerNorm::new(C_HIDDEN[0], vb.pp("embedding.1"))?;
        let embedding_conv = candle_nn::conv2d(
            c_in * patch_size * patch_size,
            C_HIDDEN[1],
            1,
            Default::default(),
            vb.pp("embedding.2"),
        )?;

        let mut down_blocks = Vec::with_capacity(C_HIDDEN.len());
        for (i, &c_hidden) in C_HIDDEN.iter().enumerate() {
            let vb = vb.pp("down_blocks").pp(i);
            let (layer_norm, conv, start_layer_i) = if i > 0 {
                let layer_norm = WLayerNorm::new(C_HIDDEN[i - 1], vb.pp(0))?;
                let cfg = candle_nn::Conv2dConfig {
                    stride: 2,
                    ..Default::default()
                };
                let conv = candle_nn::conv2d(C_HIDDEN[i - 1], c_hidden, 2, cfg, vb.pp(1))?;
                (Some(layer_norm), Some(conv), 2)
            } else {
                (None, None, 0)
            };
            let mut sub_blocks = Vec::with_capacity(BLOCKS[i]);
            let mut layer_i = start_layer_i;
            for j in 0..BLOCKS[i] {
                let c_skip = if INJECT_EFFNET[i] { c_cond } else { 0 };
                let res_block = ResBlockStageB::new(c_hidden, c_skip, 3, vb.pp(layer_i))?;
                layer_i += 1;
                let ts_block = TimestepBlock::new(c_hidden, c_r, vb.pp(layer_i))?;
                layer_i += 1;
                let attn_block = if j == 0 {
                    None
                } else {
                    let attn_block =
                        AttnBlock::new(c_hidden, c_cond, NHEAD[i], true, vb.pp(layer_i))?;
                    layer_i += 1;
                    Some(attn_block)
                };
                let sub_block = SubBlock {
                    res_block,
                    ts_block,
                    attn_block,
                };
                sub_blocks.push(sub_block)
            }
            let down_block = DownBlock {
                layer_norm,
                conv,
                sub_blocks,
            };
            down_blocks.push(down_block)
        }

        // TODO: populate.
        let up_blocks = Vec::with_capacity(C_HIDDEN.len());

        let clf_ln = WLayerNorm::new(C_HIDDEN[0], vb.pp("clf.0"))?;
        let clf_conv = candle_nn::conv2d(
            C_HIDDEN[0],
            2 * c_out * patch_size * patch_size,
            1,
            Default::default(),
            vb.pp("clf.1"),
        )?;
        Ok(Self {
            clip_mapper,
            effnet_mappers,
            seq_norm,
            embedding_conv,
            embedding_ln,
            down_blocks,
            up_blocks,
            clf_ln,
            clf_conv,
            c_r,
        })
    }

    fn gen_r_embedding(&self, r: &Tensor) -> Result<Tensor> {
        const MAX_POSITIONS: usize = 10000;
        let r = (r * MAX_POSITIONS as f64)?;
        let half_dim = self.c_r / 2;
        let emb = (MAX_POSITIONS as f64).ln() / (half_dim - 1) as f64;
        let emb = (Tensor::arange(0u32, half_dim as u32, r.device())?.to_dtype(DType::F32)?
            * -emb)?
            .exp()?;
        let emb = r.unsqueeze(1)?.broadcast_mul(&emb.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[emb.sin()?, emb.cos()?], 1)?;
        let emb = if self.c_r % 2 == 1 {
            emb.pad_with_zeros(D::Minus1, 0, 1)?
        } else {
            emb
        };
        emb.to_dtype(r.dtype())
    }

    fn gen_c_embeddings(&self, clip: &Tensor) -> Result<Tensor> {
        clip.apply(&self.clip_mapper)?.apply(&self.seq_norm)
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        r: &Tensor,
        effnet: &Tensor,
        clip: Option<&Tensor>,
    ) -> Result<Tensor> {
        const EPS: f64 = 1e-3;

        let r_embed = self.gen_r_embedding(r)?;
        let clip = match clip {
            None => None,
            Some(clip) => Some(self.gen_c_embeddings(clip)?),
        };
        let x_in = xs;

        // TODO: pixel unshuffle.
        let xs = xs.apply(&self.embedding_conv)?.apply(&self.embedding_ln)?;
        // TODO: down blocks
        let level_outputs = xs.clone();
        // TODO: up blocks
        let xs = level_outputs;
        // TODO: pxel shuffle
        let ab = xs.apply(&self.clf_ln)?.apply(&self.clf_conv)?.chunk(1, 2)?;
        let b = ((candle_nn::ops::sigmoid(&ab[1])? * (1. - EPS * 2.))? + EPS)?;
        (x_in - &ab[0])? / b
    }
}
