//! Based on the BEIT vision-language model.
//!
//! See "BEIT: BERT Pre-Training of Image Transformers", Bao et al. 2021
//! - [Arxiv](https://arxiv.org/abs/2106.08254)
//! - [Github](https://github.com/microsoft/unilm/tree/master/beit)
//!

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

const IMG_SIZE: usize = 384;
const PATCH_SIZE: usize = 16;
const NUM_CLASSES: usize = 1000;
const WINDOW_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 384 / 16 = 24
const NB_TOKENS: usize = WINDOW_SIZE * WINDOW_SIZE + 1; // 24 * 24 + 1 = 577

fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Debug)]
struct Attention {
    qkv: Linear,
    proj: Linear,
    relative_position_bias_table: Tensor,
    relative_position_index: Tensor,
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
    ) -> Result<Self> {
        let qkv = linear(vb.pp("qkv"), dim, dim * 3, qkv_bias)?;
        let proj = linear(vb.pp("proj"), dim, dim, proj_bias)?;
        // num_relative_distance = token-token(47x47) + token-CLS(1) + CLS-token(1) + CLS-CLS(1) = 2212
        let num_relative_distance = (2 * WINDOW_SIZE - 1) * (2 * WINDOW_SIZE - 1) + 3;
        let relative_position_bias_table = vb.get(
            (num_relative_distance, num_heads),
            "relative_position_bias_table",
        )?;
        let relative_position_index =
            Self::gen_relative_position_index(relative_position_bias_table.device())?;
        let scale = 1. / ((dim / num_heads) as f64).sqrt();
        Ok(Self {
            qkv,
            proj,
            relative_position_bias_table,
            relative_position_index,
            num_heads,
            scale,
        })
    }
}

impl Attention {
    // See: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/beit.py#L61
    fn gen_relative_position_index(device: &Device) -> Result<Tensor> {
        let num_relative_distance = (2 * WINDOW_SIZE - 1) * (2 * WINDOW_SIZE - 1) + 3;
        let w_area = WINDOW_SIZE * WINDOW_SIZE;

        let t_arange: Tensor = Tensor::arange(0, WINDOW_SIZE as u32, device)?;
        let t_ndgrid = Tensor::meshgrid(&[&t_arange, &t_arange], false)?;
        let coords_flatten = Tensor::stack(&t_ndgrid, 0)?.flatten(1, 2)?;

        let tmp1 = coords_flatten
            .unsqueeze(2)?
            .broadcast_as((2, w_area, w_area))?
            .to_dtype(DType::I64)?;
        let tmp2 = coords_flatten
            .unsqueeze(1)?
            .broadcast_as((2, w_area, w_area))?
            .to_dtype(DType::I64)?;
        let relative_coords = (tmp1 - tmp2)?
            .transpose(0, 1)? // 102
            .transpose(1, 2)? // 120
            .contiguous()?;

        let relative_coords = relative_coords.slice_assign(
            &[0..w_area, 0..w_area, 0..1],
            &(relative_coords.i((0..w_area, 0..w_area, 0..1))? + (WINDOW_SIZE - 1) as f64)?,
        )?;
        let relative_coords = relative_coords.slice_assign(
            &[0..w_area, 0..w_area, 1..2],
            &(relative_coords.i((0..w_area, 0..w_area, 1..2))? + (WINDOW_SIZE - 1) as f64)?,
        )?;
        let relative_coords = relative_coords.slice_assign(
            &[0..w_area, 0..w_area, 0..1],
            &(relative_coords.i((.., .., 0..1))? * (2. * (WINDOW_SIZE as f64) - 1.))?,
        )?;

        Tensor::zeros((w_area + 1, w_area + 1), DType::I64, device)?
            .slice_assign(&[1.., 1..], &relative_coords.sum(2)?)?
            .slice_assign(
                &[0..1, 0..(w_area + 1)],
                &(Tensor::ones((1, w_area + 1), DType::I64, device)?
                    * ((num_relative_distance - 3) as f64))?
                    .to_dtype(DType::I64)?,
            )?
            .slice_assign(
                &[0..(w_area + 1), 0..1],
                &(Tensor::ones((w_area + 1, 1), DType::I64, device)?
                    * ((num_relative_distance - 2) as f64))?
                    .to_dtype(DType::I64)?,
            )?
            .slice_assign(
                &[0..1, 0..1],
                &(Tensor::ones((1, 1), DType::I64, device)?
                    * ((num_relative_distance - 1) as f64))?
                    .to_dtype(DType::I64)?,
            )
    }

    fn _get_rel_pos_bias(&self) -> Result<Tensor> {
        self.relative_position_bias_table
            .index_select(
                &self
                    .relative_position_index
                    .flatten_all()?
                    .to_dtype(DType::U32)?,
                0,
            )?
            .reshape((NB_TOKENS, NB_TOKENS, ()))?
            .transpose(0, 1)? // 102
            .transpose(0, 2)? // 201
            .contiguous()?
            .unsqueeze(0)
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)? // 02134
            .transpose(0, 1)? // 20134
            .transpose(2, 3)?; // 20314
        let q = (qkv.i(0)? * self.scale)?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;
        let attn = (&q.matmul(&k.t()?)? + self._get_rel_pos_bias())?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let attn = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.proj.forward(&attn)
    }
}

#[derive(Debug)]
struct LayerScale {
    gamma: Tensor,
}

impl LayerScale {
    fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self { gamma })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.gamma)
    }
}

#[derive(Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder, in_features: usize, hidden_features: usize, bias: bool) -> Result<Self> {
        let out_features = in_features;
        let fc1 = linear(vb.pp("fc1"), in_features, hidden_features, bias)?;
        let fc2 = linear(vb.pp("fc2"), hidden_features, out_features, bias)?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.gelu()?;
        self.fc2.forward(&xs)
    }
}

#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl Block {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-6, vb.pp("norm1"))?;
        let attn = Attention::new(vb.pp("attn"), dim, num_heads, true, true)?;
        let ls1 = LayerScale::new(vb.pp("ls1"), dim)?;
        let norm2 = layer_norm(dim, 1e-6, vb.pp("norm2"))?;
        let mlp = Mlp::new(vb.pp("mlp"), dim, dim * 4, true)?;
        let ls2 = LayerScale::new(vb.pp("ls2"), dim)?;
        Ok(Self {
            norm1,
            attn,
            ls1,
            norm2,
            mlp,
            ls2,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self
            .ls1
            .forward(&self.attn.forward(&self.norm1.forward(xs)?)?)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .ls2
            .forward(&self.mlp.forward(&self.norm2.forward(&xs)?)?)?;
        xs + residual
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: candle_nn::Conv2d,
    patch_size: (usize, usize),
}

impl PatchEmbed {
    fn new(vb: VarBuilder, patch_size: usize, in_chans: usize, embed_dim: usize) -> Result<Self> {
        let config = candle_nn::Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_chans, embed_dim, patch_size, config, vb.pp("proj"))?;
        Ok(Self {
            proj,
            patch_size: (patch_size, patch_size),
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
pub struct BeitVisionTransformer {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: Linear,
}

impl BeitVisionTransformer {
    pub fn new(vb: VarBuilder, depth: usize, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let patch_embed = PatchEmbed::new(vb.pp("patch_embed"), PATCH_SIZE, 3, embed_dim)?;
        let cls_token = vb.get((1, 1, embed_dim), "cls_token")?;
        let head = linear(vb.pp("head"), embed_dim, NUM_CLASSES, true)?;
        let norm = layer_norm(embed_dim, 1e-6, vb.pp("norm"))?;
        let vb_b = vb.pp("blocks");
        let blocks = (0..depth)
            .map(|i| Block::new(vb_b.pp(i.to_string()), embed_dim, num_heads))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embed,
            cls_token,
            blocks,
            norm,
            head,
        })
    }

    fn prepare_tokens_with_mask(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.patch_embed.forward(xs)?;
        Tensor::cat(&[&self.cls_token, &xs], 1)
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

impl Module for BeitVisionTransformer {
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

pub fn vit_base(vb: VarBuilder) -> Result<BeitVisionTransformer> {
    BeitVisionTransformer::new(vb, 12, 768, 12)
}

pub fn vit_large(vb: VarBuilder) -> Result<BeitVisionTransformer> {
    BeitVisionTransformer::new(vb, 24, 1024, 16)
}
