//! Würstchen Efficient Diffusion Model
//!
//! Würstchen is an efficient diffusion model architecture for generating images using
//! a two-stage approach with a small decoder and prior network.
//!
//! - [Paper Link](https://openreview.net/pdf?id=gU58AyJlYz)
//! - [GH Link](https://github.com/dome272/Wuerstchen)
//! - [Reference Implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wuerstchen/pipeline_wuerstchen.py)
//!
pub mod attention_processor;
pub mod common;
pub mod ddpm;
pub mod diffnext;
pub mod paella_vq;
pub mod prior;
