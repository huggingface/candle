//! SAM: Segment Anything Model
//! https://github.com/facebookresearch/segment-anything
#![allow(unused)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub mod model_image_encoder;
pub mod model_mask_decoder;
pub mod model_prompt_encoder;
pub mod model_sam;
pub mod model_transformer;

use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};
use clap::Parser;

pub fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Debug)]
pub struct LayerNorm2d {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm2d {
    pub fn new(num_channels: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(num_channels, "weight")?;
        let bias = vb.get(num_channels, "bias")?;
        Ok(Self { weight, bias, eps })
    }
}

impl Module for LayerNorm2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let u = xs.mean_keepdim(1)?;
        let xs = xs.broadcast_sub(&u)?;
        let s = xs.sqr()?.mean_keepdim(1)?;
        let xs = xs.broadcast_div(&(s + self.eps)?.sqrt()?)?;
        xs.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

#[derive(Debug)]
pub struct MlpBlock {
    lin1: Linear,
    lin2: Linear,
}

impl MlpBlock {
    pub fn new(embedding_dim: usize, mlp_dim: usize, vb: VarBuilder) -> Result<Self> {
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
    model: String,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;

    let image = candle_examples::imagenet::load_image224(args.image)?.to_device(&device)?;
    println!("loaded image {image:?}");

    let weights = unsafe { candle::safetensors::MmapedFile::new(args.model)? };
    let weights = weights.deserialize()?;
    let vb = VarBuilder::from_safetensors(vec![weights], DType::F32, &device);
    let sam = model_sam::Sam::new(768, 12, 12, &[2, 5, 8, 11], vb)?; // sam_vit_b

    let (mask, iou_predictions) = sam.forward(&image, false)?;
    println!("mask: {mask:?}");
    println!("iou_predictions: {iou_predictions:?}");
    Ok(())
}
