use super::common::{AttnBlock, GlobalResponseNorm, LayerNormNoWeights, TimestepBlock, WLayerNorm};
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
        let norm = WLayerNorm::new(c)?;
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
            .contiguous()?
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
    conv: Option<candle_nn::ConvTranspose2d>,
}

#[derive(Debug)]
pub struct WDiffNeXt {
    clip_mapper: candle_nn::Linear,
    effnet_mappers: Vec<Option<candle_nn::Conv2d>>,
    seq_norm: LayerNormNoWeights,
    embedding_conv: candle_nn::Conv2d,
    embedding_ln: WLayerNorm,
    down_blocks: Vec<DownBlock>,
    up_blocks: Vec<UpBlock>,
    clf_ln: WLayerNorm,
    clf_conv: candle_nn::Conv2d,
    c_r: usize,
    patch_size: usize,
}

impl WDiffNeXt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c_in: usize,
        c_out: usize,
        c_r: usize,
        c_cond: usize,
        clip_embd: usize,
        patch_size: usize,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        const C_HIDDEN: [usize; 4] = [320, 640, 1280, 1280];
        const BLOCKS: [usize; 4] = [4, 4, 14, 4];
        const NHEAD: [usize; 4] = [1, 10, 20, 20];
        const INJECT_EFFNET: [bool; 4] = [false, true, true, true];
        const EFFNET_EMBD: usize = 16;

        let clip_mapper = candle_nn::linear(clip_embd, c_cond, vb.pp("clip_mapper"))?;
        let mut effnet_mappers = Vec::with_capacity(2 * INJECT_EFFNET.len());
        let vb_e = vb.pp("effnet_mappers");
        for (i, &inject) in INJECT_EFFNET.iter().enumerate() {
            let c = if inject {
                Some(candle_nn::conv2d(
                    EFFNET_EMBD,
                    c_cond,
                    1,
                    Default::default(),
                    vb_e.pp(i),
                )?)
            } else {
                None
            };
            effnet_mappers.push(c)
        }
        for (i, &inject) in INJECT_EFFNET.iter().rev().enumerate() {
            let c = if inject {
                Some(candle_nn::conv2d(
                    EFFNET_EMBD,
                    c_cond,
                    1,
                    Default::default(),
                    vb_e.pp(i + INJECT_EFFNET.len()),
                )?)
            } else {
                None
            };
            effnet_mappers.push(c)
        }
        let seq_norm = LayerNormNoWeights::new(c_cond)?;
        let embedding_ln = WLayerNorm::new(C_HIDDEN[0])?;
        let embedding_conv = candle_nn::conv2d(
            c_in * patch_size * patch_size,
            C_HIDDEN[0],
            1,
            Default::default(),
            vb.pp("embedding.1"),
        )?;

        let mut down_blocks = Vec::with_capacity(C_HIDDEN.len());
        for (i, &c_hidden) in C_HIDDEN.iter().enumerate() {
            let vb = vb.pp("down_blocks").pp(i);
            let (layer_norm, conv, start_layer_i) = if i > 0 {
                let layer_norm = WLayerNorm::new(C_HIDDEN[i - 1])?;
                let cfg = candle_nn::Conv2dConfig {
                    stride: 2,
                    ..Default::default()
                };
                let conv = candle_nn::conv2d(C_HIDDEN[i - 1], c_hidden, 2, cfg, vb.pp("0.1"))?;
                (Some(layer_norm), Some(conv), 1)
            } else {
                (None, None, 0)
            };
            let mut sub_blocks = Vec::with_capacity(BLOCKS[i]);
            let mut layer_i = start_layer_i;
            for _j in 0..BLOCKS[i] {
                let c_skip = if INJECT_EFFNET[i] { c_cond } else { 0 };
                let res_block = ResBlockStageB::new(c_hidden, c_skip, 3, vb.pp(layer_i))?;
                layer_i += 1;
                let ts_block = TimestepBlock::new(c_hidden, c_r, vb.pp(layer_i))?;
                layer_i += 1;
                let attn_block = if i == 0 {
                    None
                } else {
                    let attn_block = AttnBlock::new(
                        c_hidden,
                        c_cond,
                        NHEAD[i],
                        true,
                        use_flash_attn,
                        vb.pp(layer_i),
                    )?;
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

        let mut up_blocks = Vec::with_capacity(C_HIDDEN.len());
        for (i, &c_hidden) in C_HIDDEN.iter().enumerate().rev() {
            let vb = vb.pp("up_blocks").pp(C_HIDDEN.len() - 1 - i);
            let mut sub_blocks = Vec::with_capacity(BLOCKS[i]);
            let mut layer_i = 0;
            for j in 0..BLOCKS[i] {
                let c_skip = if INJECT_EFFNET[i] { c_cond } else { 0 };
                let c_skip_res = if i < BLOCKS.len() - 1 && j == 0 {
                    c_hidden + c_skip
                } else {
                    c_skip
                };
                let res_block = ResBlockStageB::new(c_hidden, c_skip_res, 3, vb.pp(layer_i))?;
                layer_i += 1;
                let ts_block = TimestepBlock::new(c_hidden, c_r, vb.pp(layer_i))?;
                layer_i += 1;
                let attn_block = if i == 0 {
                    None
                } else {
                    let attn_block = AttnBlock::new(
                        c_hidden,
                        c_cond,
                        NHEAD[i],
                        true,
                        use_flash_attn,
                        vb.pp(layer_i),
                    )?;
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
            let (layer_norm, conv) = if i > 0 {
                let layer_norm = WLayerNorm::new(C_HIDDEN[i - 1])?;
                let cfg = candle_nn::ConvTranspose2dConfig {
                    stride: 2,
                    ..Default::default()
                };
                let conv = candle_nn::conv_transpose2d(
                    c_hidden,
                    C_HIDDEN[i - 1],
                    2,
                    cfg,
                    vb.pp(layer_i).pp(1),
                )?;
                (Some(layer_norm), Some(conv))
            } else {
                (None, None)
            };
            let up_block = UpBlock {
                layer_norm,
                conv,
                sub_blocks,
            };
            up_blocks.push(up_block)
        }

        let clf_ln = WLayerNorm::new(C_HIDDEN[0])?;
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
            patch_size,
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

        let mut xs = xs
            .apply(&|xs: &_| candle_nn::ops::pixel_unshuffle(xs, self.patch_size))?
            .apply(&self.embedding_conv)?
            .apply(&self.embedding_ln)?;

        let mut level_outputs = Vec::new();
        for (i, down_block) in self.down_blocks.iter().enumerate() {
            if let Some(ln) = &down_block.layer_norm {
                xs = xs.apply(ln)?
            }
            if let Some(conv) = &down_block.conv {
                xs = xs.apply(conv)?
            }
            let skip = match &self.effnet_mappers[i] {
                None => None,
                Some(m) => {
                    let effnet = effnet.interpolate2d(xs.dim(D::Minus2)?, xs.dim(D::Minus1)?)?;
                    Some(m.forward(&effnet)?)
                }
            };
            for block in down_block.sub_blocks.iter() {
                xs = block.res_block.forward(&xs, skip.as_ref())?;
                xs = block.ts_block.forward(&xs, &r_embed)?;
                if let Some(attn_block) = &block.attn_block {
                    xs = attn_block.forward(&xs, clip.as_ref().unwrap())?;
                }
            }
            level_outputs.push(xs.clone())
        }
        level_outputs.reverse();
        let mut xs = level_outputs[0].clone();

        for (i, up_block) in self.up_blocks.iter().enumerate() {
            let effnet_c = match &self.effnet_mappers[self.down_blocks.len() + i] {
                None => None,
                Some(m) => {
                    let effnet = effnet.interpolate2d(xs.dim(D::Minus2)?, xs.dim(D::Minus1)?)?;
                    Some(m.forward(&effnet)?)
                }
            };
            for (j, block) in up_block.sub_blocks.iter().enumerate() {
                let skip = if j == 0 && i > 0 {
                    Some(&level_outputs[i])
                } else {
                    None
                };
                let skip = match (skip, effnet_c.as_ref()) {
                    (Some(skip), Some(effnet_c)) => Some(Tensor::cat(&[skip, effnet_c], 1)?),
                    (None, Some(skip)) | (Some(skip), None) => Some(skip.clone()),
                    (None, None) => None,
                };
                xs = block.res_block.forward(&xs, skip.as_ref())?;
                xs = block.ts_block.forward(&xs, &r_embed)?;
                if let Some(attn_block) = &block.attn_block {
                    xs = attn_block.forward(&xs, clip.as_ref().unwrap())?;
                }
            }
            if let Some(ln) = &up_block.layer_norm {
                xs = xs.apply(ln)?
            }
            if let Some(conv) = &up_block.conv {
                xs = xs.apply(conv)?
            }
        }

        let ab = xs
            .apply(&self.clf_ln)?
            .apply(&self.clf_conv)?
            .apply(&|xs: &_| candle_nn::ops::pixel_shuffle(xs, self.patch_size))?
            .chunk(2, 1)?;
        let b = ((candle_nn::ops::sigmoid(&ab[1])? * (1. - EPS * 2.))? + EPS)?;
        (x_in - &ab[0])? / b
    }
}
