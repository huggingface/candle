//! EVA-2 inference implementation.
//!
//! EVA-02 is a computer vision model that can be used as an ImageNet classifier.
//! The model returns the probability for an image to belong to each of the 1000
//! ImageNet categories.
//!
//! - [Paper](https://arxiv.org/abs/2303.11331). EVA-02: A Visual Representation for Neon Genesis
//! - [Code](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/eva2.py)
//!
//! # Example
//!
//! ```bash
//! cargo run \
//!   --example eva2 \
//!   --release -- \
//!   --image candle-examples/examples/yolo-v8/assets/bike.jpg
//!
//! > mountain bike, all-terrain bike, off-roader: 37.09%
//! > maillot                 : 8.30%
//! > alp                     : 2.13%
//! > bicycle-built-for-two, tandem bicycle, tandem: 0.84%
//! > crash helmet            : 0.73%
//! ```
//!
//! <div align=center>
//!   <img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/yolo-v8/assets/bike.jpg" alt="" width=640>
//! </div>
//!
use candle::{IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

const IMG_SIZE: usize = 448;
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
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    proj: Linear,
    rot_pos_embed: Tensor,
    num_heads: usize,
    scale: f64,
}

impl Attention {
    fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        proj_bias: bool,
        rot_pos_embed: &Tensor,
    ) -> Result<Self> {
        let q = linear(vb.pp("q_proj"), dim, dim, qkv_bias)?;
        let k = linear(vb.pp("k_proj"), dim, dim, false)?; // no bias for Key
        let v = linear(vb.pp("v_proj"), dim, dim, qkv_bias)?;
        let proj = linear(vb.pp("proj"), dim, dim, proj_bias)?;
        let rot_pos_embed = rot_pos_embed.clone();
        let scale = 1. / ((dim / num_heads) as f64).sqrt();
        Ok(Self {
            q,
            k,
            v,
            proj,
            rot_pos_embed,
            num_heads,
            scale,
        })
    }
}

impl Attention {
    // See: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed_sincos.py#L210
    fn apply_rot_embed_cat(x: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let cos_emb = emb.i((0.., 64..128))?; //.transpose(0, 1)?;
        let sin_emb = emb.i((0.., 0..64))?; //.transpose(0, 1)?;
        let index_even: [u32; 32] = (0u32..=63)
            .step_by(2)
            .collect::<Vec<_>>()
            .try_into()
            .expect("wrong size iterator");
        let index_odd: [u32; 32] = (1u32..=63)
            .step_by(2)
            .collect::<Vec<_>>()
            .try_into()
            .expect("wrong size iterator");
        let t_index_even = Tensor::new(&index_even, x.device())?;
        let t_index_odd = Tensor::new(&index_odd, x.device())?;
        let x_c = x.contiguous()?;
        let rot_x_even = x_c.index_select(&t_index_even, D::Minus1)?;
        let rot_x_odd_minus = (-1.0 * x_c.index_select(&t_index_odd, D::Minus1)?)?;
        let rot_x =
            Tensor::stack(&[&rot_x_odd_minus, &rot_x_even], D::Minus1)?.reshape(x.shape())?;
        x.broadcast_mul(&cos_emb)? + rot_x.broadcast_mul(&sin_emb)?
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = Tensor::cat(
            &[
                &self.q.forward(xs)?,
                &self.k.forward(xs)?,
                &self.v.forward(xs)?,
            ],
            2,
        )?
        .reshape((b, n, 3, self.num_heads, c / self.num_heads))?
        .transpose(1, 2)? // 02134
        .transpose(0, 1)? // 20134
        .transpose(2, 3)?; // 20314
        let q = qkv.i(0)?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;

        let npt = 1; // num_prefix_tokens = 1 for CLS token
        let q = Tensor::cat(
            &[
                &q.i((0.., 0.., ..npt, 0..))?,
                &Self::apply_rot_embed_cat(&q.i((0.., 0.., npt.., 0..))?, &self.rot_pos_embed)?,
            ],
            2,
        )?;
        let k = Tensor::cat(
            &[
                &k.i((0.., 0.., ..npt, 0..))?,
                &Self::apply_rot_embed_cat(&k.i((0.., 0.., npt.., 0..))?, &self.rot_pos_embed)?,
            ],
            2,
        )?;

        let q = (q * self.scale)?;
        let attn = &q.matmul(&k.t()?)?;
        let attn = candle_nn::ops::softmax(attn, D::Minus1)?;
        let attn = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.proj.forward(&attn)
    }
}

#[derive(Debug)]
struct Mlp {
    fc1_g: Linear,
    fc1_x: Linear,
    norm: LayerNorm,
    fc2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder, in_features: usize, hidden_features: usize, bias: bool) -> Result<Self> {
        let out_features = in_features;
        let fc1_g = linear(vb.pp("fc1_g"), in_features, hidden_features, bias)?;
        let fc1_x = linear(vb.pp("fc1_x"), in_features, hidden_features, bias)?;
        let norm = layer_norm(hidden_features, 1e-6, vb.pp("norm"))?;
        let fc2 = linear(vb.pp("fc2"), hidden_features, out_features, bias)?;
        Ok(Self {
            fc1_g,
            fc1_x,
            norm,
            fc2,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_g = self.fc1_g.forward(xs)?.silu()?;
        let xs = self.fc1_x.forward(xs)?;
        let xs = self.norm.forward(&(xs_g.mul(&xs)?))?;
        self.fc2.forward(&xs)
    }
}

#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize, rot_pos_embed: &Tensor) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-6, vb.pp("norm1"))?;
        let attn = Attention::new(vb.pp("attn"), dim, num_heads, true, true, rot_pos_embed)?;
        let norm2 = layer_norm(dim, 1e-6, vb.pp("norm2"))?;
        let hidden_dim = dim * 4 * 2 / 3; // 768 * 4 * 2 / 3 = 3072 * 2 / 3 = 2048
        let mlp = Mlp::new(vb.pp("mlp"), dim, hidden_dim, true)?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = &self.attn.forward(&self.norm1.forward(xs)?)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = &self.mlp.forward(&self.norm2.forward(&xs)?)?;
        xs + residual
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: candle_nn::Conv2d,
    patch_size: (usize, usize),
    num_patches: usize,
}

impl PatchEmbed {
    fn new(
        vb: VarBuilder,
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
    ) -> Result<Self> {
        let config = candle_nn::Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_chans, embed_dim, patch_size, config, vb.pp("proj"))?;
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        Ok(Self {
            proj,
            patch_size: (patch_size, patch_size),
            num_patches,
        })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        let (patch_h, patch_w) = self.patch_size;
        if (h % patch_h) != 0 {
            candle::bail!("image height {h} is not a multiple of patch height {patch_h}")
        }
        if (w % patch_w) != 0 {
            candle::bail!("image width {w} is not a multiple of patch width {patch_w}")
        }
        let xs = self.proj.forward(xs)?;
        let (b, c, h, w) = xs.dims4()?;
        // flatten embeddings.
        xs.reshape((b, c, h * w))?.transpose(1, 2)
    }
}

#[derive(Debug)]
pub struct EVA2VisionTransformer {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: Linear,
}

impl EVA2VisionTransformer {
    pub fn new(vb: VarBuilder, depth: usize, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let patch_embed =
            PatchEmbed::new(vb.pp("patch_embed"), IMG_SIZE, PATCH_SIZE, 3, embed_dim)?;
        let cls_token = vb.get((1, 1, embed_dim), "cls_token")?;
        let pos_embed = vb.get((1, patch_embed.num_patches + 1, embed_dim), "pos_embed")?;
        let rot_pos_embed = vb.get((patch_embed.num_patches, 128), "rot_pos_embed")?;
        let head = linear(vb.pp("head"), embed_dim, NUM_CLASSES, true)?;
        let norm = layer_norm(embed_dim, 1e-6, vb.pp("norm"))?;
        let vb_b = vb.pp("blocks");
        let blocks = (0..depth)
            .map(|i| Block::new(vb_b.pp(i.to_string()), embed_dim, num_heads, &rot_pos_embed))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
            head,
        })
    }

    fn interpolate_pos_encoding(
        &self,
        xs: &Tensor,
        w: usize,
        h: usize,
        num_prefix_tokens: usize,
    ) -> Result<Tensor> {
        let npatch = xs.dim(1)? - 1;
        let n = self.pos_embed.dim(1)? - 1;
        let sqrt_n = (n as f64).sqrt();
        if npatch == n && w == h {
            return Ok(self.pos_embed.clone());
        }
        // Interpolate only local tokens, i.e. those after the CLS token
        let prefix_tokens_pos_embed = self.pos_embed.i((0.., ..num_prefix_tokens, 0..))?.clone();
        let patch_pos_embed = &self.pos_embed.i((0.., num_prefix_tokens.., 0..))?;
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
        Tensor::cat(&[&prefix_tokens_pos_embed, &patch_pos_embed], 1)
    }

    fn prepare_tokens_with_mask(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _nc, w, h) = xs.dims4()?;
        if (w != IMG_SIZE) || (h != IMG_SIZE) {
            panic!("Error: The input tensor should have the shape: Bx3x518x518.");
        }
        let xs = self.patch_embed.forward(xs)?;
        let xs = Tensor::cat(&[&self.cls_token, &xs], 1)?;
        let xs = (&xs + &self.interpolate_pos_encoding(&xs, w, h, 1)?)?;
        Ok(xs)
    }

    fn get_intermediate_layers_not_chunked(
        &self,
        xs: &Tensor,
        blocks_to_take: &[usize],
    ) -> Result<Vec<Tensor>> {
        let mut xs = self.prepare_tokens_with_mask(xs)?;
        let mut output = Vec::new();
        for (i, blk) in self.blocks.iter().enumerate() {
            xs = blk.forward(&xs)?;
            if blocks_to_take.contains(&i) {
                output.push(xs.clone());
            }
        }
        if output.len() != blocks_to_take.len() {
            candle::bail!(
                "only {} / {} blocks found",
                output.len(),
                blocks_to_take.len()
            );
        }
        Ok(output)
    }

    pub fn get_intermediate_layers(
        &self,
        xs: &Tensor,
        blocks_to_take: &[usize],
        reshape: bool,
        return_class_token: bool,
        norm: bool,
    ) -> Result<Tensor> {
        let outputs = self.get_intermediate_layers_not_chunked(xs, blocks_to_take)?;
        let outputs = if norm {
            outputs
                .iter()
                .map(|out| self.norm.forward(out))
                .collect::<Result<Vec<_>>>()?
        } else {
            outputs
        };
        let class_tokens = outputs
            .iter()
            .map(|out| out.i((.., 0)))
            .collect::<Result<Vec<_>>>()?;
        let outputs = outputs
            .iter()
            .map(|out| out.i((.., 1..)))
            .collect::<Result<Vec<_>>>()?;

        let outputs = if reshape {
            let (b, _c, w, h) = xs.dims4()?;
            let patch_size = self.patch_embed.patch_size.0;
            let num_channels = outputs[0].elem_count() / (b * (w / patch_size) * (h / patch_size));
            outputs
                .iter()
                .map(|out| {
                    out.reshape((b, w / patch_size, h / patch_size, num_channels))?
                        .transpose(2, 3)?
                        .transpose(1, 2)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            outputs
        };

        let outputs = if return_class_token {
            outputs
                .iter()
                .zip(class_tokens.iter())
                .map(|(out, class_token)| Tensor::cat(&[out, class_token], D::Minus1))
                .collect::<Result<Vec<_>>>()?
        } else {
            outputs
        };

        Tensor::stack(&outputs[..], 0)
    }
}

impl Module for EVA2VisionTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.prepare_tokens_with_mask(xs)?;
        for blk in self.blocks.iter() {
            xs = blk.forward(&xs)?
        }
        let xs_moy_local_tokens = xs.i((.., 1..))?.mean(1)?;
        let xs_norm = self.norm.forward(&xs_moy_local_tokens)?;
        self.head.forward(&xs_norm)
    }
}

pub fn vit_base(vb: VarBuilder) -> Result<EVA2VisionTransformer> {
    EVA2VisionTransformer::new(vb, 12, 768, 12)
}

pub fn vit_large(vb: VarBuilder) -> Result<EVA2VisionTransformer> {
    EVA2VisionTransformer::new(vb, 24, 1024, 16)
}
