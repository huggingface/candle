//! BiRefNet (Bilateral Reference Network) for High-Resolution Image Segmentation
//!
//! BiRefNet is a deep learning model designed for high-quality image background removal
//! and binary image segmentation tasks.
//!
//! ## Architecture Overview
//!
//! The model consists of:
//! - Swin Transformer backbone for feature extraction
//! - Squeeze module for channel reduction
//! - Decoder with ASPPDeformable attention for multi-scale feature fusion
//! - Multi-scale supervision heads
//!
//! ## Example Usage
//!
//! ```ignore
//! use candle::{Device, DType};
//! use candle_transformers::models::birefnet::{BiRefNet, Config};
//!
//! let device = Device::Cpu;
//! let model = BiRefNet::load("model.safetensors", &device)?;
//! let output = model.forward(&input)?;
//! ```

mod aspp;
#[allow(clippy::module_inception)]
mod birefnet;
mod blocks;
mod config;
mod decoder;
mod swin;

pub use aspp::{ASPPDeformable, DeformableConv2d, GlobalAvgPool, ASPP};
pub use birefnet::BiRefNet;
pub use blocks::{BasicDecBlk, BasicLatBlk, SimpleConvs};
pub use config::{
    BackboneType, Config, DecoderAttentionType, DecoderBlockType, MultiScaleInputType,
    SqueezeBlockType,
};
pub use decoder::Decoder;
pub use swin::{BasicLayer, PatchEmbed, PatchMerging, SwinTransformer, SwinTransformerBlock};
