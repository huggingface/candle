//! SAM: Segment Anything Model
//! https://github.com/facebookresearch/segment-anything
#![allow(unused)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;

use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

const IMG_SIZE: usize = 518;
const PATCH_SIZE: usize = 14;
const NUM_CLASSES: usize = 1000;

fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Debug)]
struct MlpBlock {
    lin1: Linear,
    lin2: Linear,
}

impl MlpBlock {
    fn new(embedding_dim: usize, mlp_dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin1 = candle_nn::linear(embedding_dim, mlp_dim, vb.pp("lin1"))?;
        let lin2 = candle_nn::linear(mlp_dim, embedding_dim, vb.pp("lin2"))?;
        Ok(Self { lin1, lin2 })
    }
}

impl Module for MlpBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.lin1)?.gelu()?.apply(&self.lin2)
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: candle_nn::Conv2d,
}

impl PatchEmbed {
    fn new(
        in_chans: usize,
        embed_dim: usize,
        k_size: usize,
        stride: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = candle_nn::Conv2dConfig {
            stride,
            padding,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_chans, embed_dim, k_size, cfg, vb.pp("proj"))?;
        Ok(Self { proj })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.proj)?.permute((0, 2, 3, 1))
    }
}

#[derive(Debug)]
struct Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    scale: f64,
    use_rel_pos: bool,
    rel_pos_hw: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        use_rel_pos: bool,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let qkv = linear(vb.pp("qkv"), dim, dim * 3, qkv_bias)?;
        let proj = linear(vb.pp("proj"), dim, dim, true)?;
        let head_dim = dim / num_heads;
        let scale = 1. / (head_dim as f64).sqrt();
        let rel_pos_hw = if use_rel_pos {
            let h = vb.get((2 * window_size - 1, head_dim), "rel_pos_h")?;
            let w = vb.get((2 * window_size - 1, head_dim), "rel_pos_w")?;
            Some((h, w))
        } else {
            None
        };
        Ok(Self {
            qkv,
            proj,
            num_heads,
            scale,
            use_rel_pos,
            rel_pos_hw,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, h, w, c) = xs.dims4()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, h * w, 3, self.num_heads, c / self.num_heads))?
            .permute((2, 0, 3, 1, 4))?
            .reshape((3, b * self.num_heads, h * w, c / self.num_heads))?;
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;
        let attn = (q * self.scale)?.matmul(&k.t()?)?;
        if self.use_rel_pos {
            todo!()
        }
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let attn = attn
            .matmul(&v)?
            .reshape((b, self.num_heads, h, w, c / self.num_heads))?
            .permute((0, 2, 3, 1, 4))?
            .reshape((b, h, w, c / self.num_heads))?;
        self.proj.forward(&attn)
    }
}

#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: MlpBlock,
    window_size: usize,
}

impl Block {
    fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        use_rel_pos: bool,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let attn = Attention::new(
            dim,
            num_heads,
            qkv_bias,
            use_rel_pos,
            window_size,
            vb.pp("attn"),
        )?;
        let mlp = MlpBlock::new(dim, dim * 4, vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shortcut = xs;
        let xs = self.norm1.forward(xs)?;
        if self.window_size > 0 {
            todo!()
        }
        let xs = self.attn.forward(&xs)?;
        if self.window_size > 0 {
            todo!()
        }
        let xs = (xs + shortcut)?;
        &xs + xs.apply(&self.norm2)?.apply(&self.mlp)?
    }
}

#[derive(Debug)]
struct ImageEncoderViT {
    img_size: usize,
    patch_embed: PatchEmbed,
    blocks: Vec<Block>,
    neck_conv1: candle_nn::Conv2d,
    neck_ln1: LayerNorm,
    neck_conv2: candle_nn::Conv2d,
    neck_ln2: LayerNorm,
    pos_embed: Option<Tensor>,
}

impl ImageEncoderViT {
    #[allow(clippy::too_many_arguments)]
    fn new(
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        out_chans: usize,
        qkv_bias: bool,
        use_rel_pos: bool,
        use_abs_pos: bool,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::new(
            in_chans,
            embed_dim,
            patch_size,
            patch_size,
            0,
            vb.pp("patch_embed"),
        )?;
        let mut blocks = Vec::with_capacity(depth);
        let vb_b = vb.pp("blocks");
        for i in 0..depth {
            let block = Block::new(
                embed_dim,
                num_heads,
                qkv_bias,
                use_rel_pos,
                window_size,
                vb_b.pp(i),
            )?;
            blocks.push(block)
        }
        let neck_conv1 = candle_nn::conv2d_no_bias(
            embed_dim,
            out_chans,
            1,
            Default::default(),
            vb.pp("neck.0"),
        )?;
        let neck_ln1 = layer_norm(out_chans, 1e-6, vb.pp("neck.1"))?;
        let cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let neck_conv2 = candle_nn::conv2d_no_bias(out_chans, out_chans, 3, cfg, vb.pp("neck.2"))?;
        let neck_ln2 = layer_norm(out_chans, 1e-6, vb.pp("neck.3"))?;
        let pos_embed = if use_abs_pos {
            let p = vb.get(
                (1, img_size / patch_size, img_size / patch_size, embed_dim),
                "pos_embed",
            )?;
            Some(p)
        } else {
            None
        };
        Ok(Self {
            img_size,
            patch_embed,
            blocks,
            neck_conv1,
            neck_ln1,
            neck_conv2,
            neck_ln2,
            pos_embed,
        })
    }
}

impl Module for ImageEncoderViT {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.patch_embed.forward(xs)?;
        let mut xs = match &self.pos_embed {
            Some(pos_embed) => (xs + pos_embed)?,
            None => xs,
        };
        for block in self.blocks.iter() {
            xs = block.forward(&xs)?
        }
        xs.permute((0, 3, 1, 2))?
            .apply(&self.neck_conv1)?
            .apply(&self.neck_ln1)?
            .apply(&self.neck_conv2)?
            .apply(&self.neck_ln2)
    }
}

#[derive(Debug)]
struct MlpMaskDecoder {
    layers: Vec<Linear>,
    sigmoid_output: bool,
}

impl MlpMaskDecoder {
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        sigmoid_output: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let vb = vb.pp("layers");
        for i in 0..num_layers {
            let in_dim = if i == 0 { input_dim } else { hidden_dim };
            let out_dim = if i + 1 == num_layers {
                output_dim
            } else {
                hidden_dim
            };
            let layer = linear(vb.pp(i), in_dim, out_dim, true)?;
            layers.push(layer)
        }
        Ok(Self {
            layers,
            sigmoid_output,
        })
    }
}

impl Module for MlpMaskDecoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs)?;
            if i + 1 < self.layers.len() {
                xs = xs.relu()?
            }
        }
        if self.sigmoid_output {
            candle_nn::ops::sigmoid(&xs)
        } else {
            Ok(xs)
        }
    }
}

#[derive(Debug)]
struct MaskDecoder {
    iou_tokens: candle_nn::Embedding,
    mask_tokens: candle_nn::Embedding,
    iou_prediction_head: MlpMaskDecoder,
}

impl MaskDecoder {
    fn new(
        transformer_dim: usize,
        num_multimask_outputs: usize,
        iou_head_depth: usize,
        iou_head_hidden_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_mask_tokens = num_multimask_outputs - 1;
        let iou_prediction_head = MlpMaskDecoder::new(
            transformer_dim,
            iou_head_hidden_dim,
            num_mask_tokens,
            iou_head_depth,
            false,
            vb.pp("iou_prediction_head"),
        )?;
        let iou_tokens = candle_nn::embedding(1, transformer_dim, vb.pp("iou_tokens"))?;
        let mask_tokens =
            candle_nn::embedding(num_mask_tokens, transformer_dim, vb.pp("mask_tokens"))?;
        Ok(Self {
            iou_tokens,
            mask_tokens,
            iou_prediction_head,
        })
    }
}

/*
    fn interpolate_pos_encoding(&self, xs: &Tensor, w: usize, h: usize) -> Result<Tensor> {
        let npatch = xs.dim(1)? - 1;
        let n = self.pos_embed.dim(1)? - 1;
        let sqrt_n = (n as f64).sqrt();
        if npatch == n && w == h {
            return Ok(xs.clone());
        }
        let class_pos_embed = self.pos_embed.i((.., ..1))?;
        let patch_pos_embed = self.pos_embed.i((.., 1..))?;
        let dim = xs.dim(D::Minus1)?;
        let (w0, h0) = ((w / PATCH_SIZE) as f64 + 0.1, (h / PATCH_SIZE) as f64 + 0.1);
        let patch_pos_embed = patch_pos_embed
            .reshape((1, sqrt_n as usize, sqrt_n as usize, dim))?
            .transpose(2, 3)?
            .transpose(1, 2)?;
        // This uses bicubic interpolation in the original implementation.
        let patch_pos_embed = patch_pos_embed.upsample_nearest2d(h0 as usize, w0 as usize)?;
        let el_count = patch_pos_embed.shape().elem_count();
        let patch_pos_embed =
            patch_pos_embed
                .transpose(1, 2)?
                .transpose(2, 3)?
                .reshape((1, el_count / dim, dim))?;
        Tensor::cat(&[&class_pos_embed, &patch_pos_embed], 1)
    }

    fn prepare_tokens_with_mask(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _nc, w, h) = xs.dims4()?;
        let xs = self.patch_embed.forward(xs)?;
        let xs = Tensor::cat(&[&self.cls_token, &xs], 1)?;
        &xs + &self.interpolate_pos_encoding(&xs, w, h)?
    }
*/

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let _device = candle_examples::device(args.cpu)?;

    let image = candle_examples::imagenet::load_image224(args.image)?;
    println!("loaded image {image:?}");

    Ok(())
}
