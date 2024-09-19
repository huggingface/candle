//! Chinese contrastive Language-Image Pre-Training
//!
//! Chinese contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
//! pairs of images with related texts.
//!
//! https://github.com/OFA-Sys/Chinese-CLIP
//! https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/chinese_clip/modeling_chinese_clip.py

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn as nn;
use candle_nn::Module;

pub mod text_model;
pub mod vision_model;

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    QuickGelu,
    Gelu,
    GeluNew,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Activation::QuickGelu => xs * nn::ops::sigmoid(&(xs * 1.702f64)?)?,
            Activation::Gelu => xs.gelu_erf(),
            Activation::GeluNew => xs.gelu(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChineseClipConfig {
    pub text_config: text_model::ChineseClipTextConfig,
    pub vision_config: vision_model::ChineseClipVisionConfig,
    pub logit_scale_init_value: f32,
    pub image_size: usize,
}

impl ChineseClipConfig {
    /// referer: https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/blob/main/config.json
    pub fn clip_vit_base_patch16() -> Self {
        let text_config = text_model::ChineseClipTextConfig::clip_vit_base_patch16();
        let vision_config = vision_model::ChineseClipVisionConfig::clip_vit_base_patch16();

        Self {
            text_config,
            vision_config,
            logit_scale_init_value: 2.6592,
            image_size: 512,
        }
    }
}
