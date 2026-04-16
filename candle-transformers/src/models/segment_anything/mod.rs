//! Segment Anything Model (SAM)
//!
//! SAM is an architecture for image segmentation, capable of segmenting any object
//! in an image based on prompts like points or boxes. //! This model provides a robust and fast image segmentation pipeline that can be tweaked via
//! some prompting (requesting some points to be in the target mask, requesting some
//! points to be part of the background so _not_ in the target mask, specifying some
//! bounding box).
//!
//! - ⚡ [Interactive Wasm Example](https://huggingface.co/spaces/radames/candle-segment-anything-wasm)
//! - 💻 [GH Link](https://github.com/facebookresearch/segment-anything)
//! - 📝 [Paper](https://arxiv.org/abs/2304.02643)
//! - 💡 The default backbone can be replaced by the smaller and faster TinyViT model based on [MobileSAM](https://github.com/ChaoningZhang/MobileSAM).
//!
//!
//! ## Example
//!
//! ```bash
//! cargo run --example segment-anything --release -- \
//!     --image candle-examples/examples/yolo-v8/assets/bike.jpg
//!     --use-tiny --point 0.6,0.6 --point 0.6,0.55
//! ```
//!
//! <div align=center style="display: flex; justify-content: center; gap: 10px;">
//!   <img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/yolo-v8/assets/bike.jpg" alt="" width="30%">
//!   <img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/segment-anything/assets/single_pt_prompt.jpg" alt="" width="30%">
//!   <img src="https://github.com/huggingface/candle/raw/main/candle-examples/examples/segment-anything/assets/two_pt_prompt.jpg" alt="" width="30%">
//! </div>
//!
//!
//! > Original; Prompt with `--point 0.6,0.55`; Prompt with `--point 0.6,0.6 --point 0.6,0.55`
//!
pub use crate::models::with_tracing::Linear;
use candle::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub mod image_encoder;
pub mod mask_decoder;
pub mod prompt_encoder;
pub mod sam;
pub mod tiny_vit;
pub mod transformer;

pub fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    if bias {
        crate::models::with_tracing::linear(in_dim, out_dim, vb)
    } else {
        crate::models::with_tracing::linear_no_bias(in_dim, out_dim, vb)
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
    span: tracing::Span,
}

impl MlpBlock {
    pub fn new(
        embedding_dim: usize,
        mlp_dim: usize,
        activation: candle_nn::Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let lin1 = linear(vb.pp("lin1"), embedding_dim, mlp_dim, true)?;
        let lin2 = linear(vb.pp("lin2"), mlp_dim, embedding_dim, true)?;
        let span = tracing::span!(tracing::Level::TRACE, "mlp-block");
        Ok(Self {
            lin1,
            lin2,
            activation,
            span,
        })
    }
}

impl Module for MlpBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        xs.apply(&self.lin1)?
            .apply(&self.activation)?
            .apply(&self.lin2)
    }
}
