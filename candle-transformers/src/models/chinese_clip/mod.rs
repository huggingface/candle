//! Chinese contrastive Language-Image Pre-Training
//!
//! Chinese contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
//! pairs of images with related texts.
//!
//! - 💻 [GH Link](https://github.com/OFA-Sys/Chinese-CLIP)
//! - 💻 Transformers Python [reference implementation](https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/chinese_clip/modeling_chinese_clip.py)
//!
use candle::{Module, Result, Tensor, D};
use candle_nn as nn;

use text_model::ChineseClipTextTransformer;
use vision_model::ChineseClipVisionTransformer;

pub mod text_model;
pub mod vision_model;

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    QuickGelu,
    Gelu,
    GeluNew,
    Relu,
}

impl From<String> for Activation {
    fn from(value: String) -> Self {
        match value.as_str() {
            "quick_gelu" => Activation::QuickGelu,
            "gelu" => Activation::Gelu,
            "gelu_new" => Activation::GeluNew,
            "relu" => Activation::Relu,
            _ => panic!("Invalid activation function: {}", value),
        }
    }
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Activation::QuickGelu => xs * nn::ops::sigmoid(&(xs * 1.702f64)?)?,
            Activation::Gelu => xs.gelu_erf(),
            Activation::GeluNew => xs.gelu(),
            Activation::Relu => xs.relu(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChineseClipConfig {
    pub text_config: text_model::ChineseClipTextConfig,
    pub vision_config: vision_model::ChineseClipVisionConfig,
    pub projection_dim: usize,
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
            projection_dim: 512,
            logit_scale_init_value: 2.6592,
            image_size: 512,
        }
    }
}

#[derive(Clone, Debug)]
pub enum EncoderConfig {
    Text(text_model::ChineseClipTextConfig),
    Vision(vision_model::ChineseClipVisionConfig),
}

impl EncoderConfig {
    pub fn embed_dim(&self) -> usize {
        match self {
            Self::Text(c) => c.hidden_size,
            Self::Vision(c) => c.hidden_size,
        }
    }

    pub fn num_attention_heads(&self) -> usize {
        match self {
            Self::Text(c) => c.num_attention_heads,
            Self::Vision(c) => c.num_attention_heads,
        }
    }

    pub fn intermediate_size(&self) -> usize {
        match self {
            Self::Text(c) => c.intermediate_size,
            Self::Vision(c) => c.intermediate_size,
        }
    }

    pub fn num_hidden_layers(&self) -> usize {
        match self {
            Self::Text(c) => c.num_hidden_layers,
            Self::Vision(c) => c.num_hidden_layers,
        }
    }

    pub fn activation(&self) -> Activation {
        match self {
            Self::Text(c) => c.hidden_act,
            Self::Vision(c) => c.hidden_act,
        }
    }

    pub fn layer_norm_eps(&self) -> f64 {
        match self {
            Self::Text(c) => c.layer_norm_eps,
            Self::Vision(c) => c.layer_norm_eps,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChineseClipModel {
    text_model: ChineseClipTextTransformer,
    vision_model: ChineseClipVisionTransformer,
    visual_projection: nn::Linear,
    text_projection: nn::Linear,
    logit_scale: Tensor,
}

impl ChineseClipModel {
    pub fn new(vs: nn::VarBuilder, c: &ChineseClipConfig) -> Result<Self> {
        let text_model = ChineseClipTextTransformer::new(vs.pp("text_model"), &c.text_config)?;

        let vision_model =
            ChineseClipVisionTransformer::new(vs.pp("vision_model"), &c.vision_config)?;

        let vision_embed_dim = c.vision_config.hidden_size;
        let vision_projection = nn::linear_no_bias(
            vision_embed_dim,
            c.projection_dim,
            vs.pp("visual_projection"),
        )?;

        let text_embed_dim = c.text_config.hidden_size;
        let text_projection =
            nn::linear_no_bias(text_embed_dim, c.projection_dim, vs.pp("text_projection"))?;

        let logit_scale = if vs.contains_tensor("logit_scale") {
            vs.get(&[], "logit_scale")?
        } else {
            Tensor::new(&[c.logit_scale_init_value], vs.device())?
        };

        Ok(Self {
            text_model,
            vision_model,
            visual_projection: vision_projection,
            text_projection,
            logit_scale,
        })
    }

    pub fn get_text_features(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let output = self
            .text_model
            .forward(input_ids, token_type_ids, attention_mask)?
            .contiguous()?;
        self.text_projection.forward(&output)
    }

    pub fn get_image_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        pixel_values
            .apply(&self.vision_model)?
            .apply(&self.visual_projection)
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let image_features = self.get_image_features(pixel_values)?;
        let text_features = self.get_text_features(input_ids, token_type_ids, attention_mask)?;

        let image_features_normalized = div_l2_norm(&image_features)?;
        let text_features_normalized = div_l2_norm(&text_features)?;

        let logits_per_text = text_features_normalized.matmul(&image_features_normalized.t()?)?;
        let logit_scale = self.logit_scale.exp()?;
        let logits_per_text = logits_per_text.broadcast_mul(&logit_scale)?;
        let logits_per_image = logits_per_text.t()?;
        Ok((logits_per_text, logits_per_image))
    }
}

pub fn div_l2_norm(v: &Tensor) -> Result<Tensor> {
    let l2_norm = v.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    v.broadcast_div(&l2_norm)
}
