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
use candle_nn::{Linear, Module, VarBuilder};
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
    num_channels: usize,
    eps: f64,
}

impl LayerNorm2d {
    pub fn new(num_channels: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(num_channels, "weight")?;
        let bias = vb.get(num_channels, "bias")?;
        Ok(Self {
            weight,
            bias,
            num_channels,
            eps,
        })
    }
}

impl Module for LayerNorm2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let u = xs.mean_keepdim(1)?;
        let xs = xs.broadcast_sub(&u)?;
        let s = xs.sqr()?.mean_keepdim(1)?;
        let xs = xs.broadcast_div(&(s + self.eps)?.sqrt()?)?;
        xs.broadcast_mul(&self.weight.reshape((1, self.num_channels, 1, 1))?)?
            .broadcast_add(&self.bias.reshape((1, self.num_channels, 1, 1))?)
    }
}

#[derive(Debug)]
pub struct MlpBlock {
    lin1: Linear,
    lin2: Linear,
    activation: candle_nn::Activation,
}

impl MlpBlock {
    pub fn new(
        embedding_dim: usize,
        mlp_dim: usize,
        activation: candle_nn::Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let lin1 = candle_nn::linear(embedding_dim, mlp_dim, vb.pp("lin1"))?;
        let lin2 = candle_nn::linear(mlp_dim, embedding_dim, vb.pp("lin2"))?;
        Ok(Self {
            lin1,
            lin2,
            activation,
        })
    }
}

impl Module for MlpBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.lin1)?
            .apply(&self.activation)?
            .apply(&self.lin2)
    }
}

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

    let device = candle_examples::device(args.cpu)?;

    let image =
        candle_examples::load_image(args.image, Some(model_sam::IMAGE_SIZE))?.to_device(&device)?;
    println!("loaded image {image:?}");

    let model = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/candle-sam".to_string());
            api.get("sam_vit_b_01ec64.safetensors")?
        }
    };
    let weights = unsafe { candle::safetensors::MmapedFile::new(model)? };
    let weights = weights.deserialize()?;
    let vb = VarBuilder::from_safetensors(vec![weights], DType::F32, &device);
    let sam = model_sam::Sam::new(768, 12, 12, &[2, 5, 8, 11], vb)?; // sam_vit_b

    let (mask, iou_predictions) = sam.forward(&image, false)?;
    println!("mask:\n{mask}");
    println!("iou_predictions: {iou_predictions:?}");
    Ok(())
}
