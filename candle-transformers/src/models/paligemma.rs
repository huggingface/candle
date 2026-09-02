//! Multimodal multi-purpose model combining Gemma-based language model with SigLIP image understanding
//!
//! See PaLiGemma details at:
//! - [Paper](https://arxiv.org/abs/2402.05257)
//! - [Google Blog Post](https://blog.research.google/2024/02/paligemma-scaling-language-image.html)
//!
//! The model is a multimodal combination of:
//! - SigLIP vision encoder
//! - Gemma language model
//! - Cross-projection layers
//!
//! References:
//! - [HuggingFace Implementation](https://huggingface.co/google/paligemma-3b)
//! - [Paper: PaLI-3 and Beyond: Scaling Language-Image Learning](https://arxiv.org/abs/2402.05257)
//!

use crate::models::{gemma, siglip};
use candle::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Config {
    pub vision_config: siglip::VisionConfig,
    pub text_config: gemma::Config,
    pub projection_dim: usize,
}

impl Config {
    pub fn paligemma_3b_224() -> Self {
        // https://huggingface.co/google/paligemma-3b-pt-224/blob/main/config.json
        Self {
            vision_config: siglip::VisionConfig::paligemma_3b_224(),
            text_config: gemma::Config {
                hidden_size: 2048,
                intermediate_size: 16384,
                num_attention_heads: 8,
                num_hidden_layers: 18,
                num_key_value_heads: 1,
                vocab_size: 257216,
                // Default values.
                rope_theta: 10000.,
                head_dim: 256,
                hidden_act: Some(candle_nn::Activation::GeluPytorchTanh),
                hidden_activation: None,
                attention_bias: false,
                max_position_embeddings: 8192,
                rms_norm_eps: 1e-6,
            },
            projection_dim: 2048,
        }
    }

    pub fn paligemma_3b_448() -> Self {
        Self {
            vision_config: siglip::VisionConfig::paligemma_3b_448(),
            text_config: gemma::Config {
                hidden_size: 2048,
                intermediate_size: 16384,
                num_attention_heads: 8,
                num_hidden_layers: 18,
                num_key_value_heads: 1,
                // Default values.
                rope_theta: 10000.,
                head_dim: 256,
                hidden_act: Some(candle_nn::Activation::GeluPytorchTanh),
                hidden_activation: None,
                attention_bias: false,
                max_position_embeddings: 8192,
                rms_norm_eps: 1e-6,
                vocab_size: 257216,
            },
            projection_dim: 2048,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MultiModalProjector {
    linear: Linear,
}

impl MultiModalProjector {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let linear = linear(
            cfg.vision_config.hidden_size,
            cfg.projection_dim,
            vb.pp("linear"),
        )?;
        Ok(Self { linear })
    }
}

impl Module for MultiModalProjector {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.linear)
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    pos: usize,
    vision_tower: siglip::VisionModel,
    multi_modal_projector: MultiModalProjector,
    language_model: gemma::Model,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vision_tower = siglip::VisionModel::new(
            &cfg.vision_config,
            false,
            vb.pp("vision_tower.vision_model"),
        )?;
        let multi_modal_projector = MultiModalProjector::new(cfg, vb.pp("multi_modal_projector"))?;
        let language_model = gemma::Model::new(false, &cfg.text_config, vb.pp("language_model"))?;
        Ok(Self {
            pos: 0,
            language_model,
            vision_tower,
            multi_modal_projector,
        })
    }

    pub fn setup(&mut self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        self.clear_kv_cache();
        let image_features = self
            .vision_tower
            .forward(pixel_values)?
            .apply(&self.multi_modal_projector)?;
        let image_features = crate::models::clip::div_l2_norm(&image_features)?;
        let text_features = self.language_model.embed_tokens().forward(input_ids)?;
        let input_embeds = Tensor::cat(&[image_features, text_features], 1)?;
        self.pos = input_embeds.dim(1)?;
        self.language_model.forward_embeds(&input_embeds, None, 0)
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let pos = self.pos;
        let seq_len = input_ids.dim(1)?;
        self.pos = pos + seq_len;
        self.language_model.forward(input_ids, pos)
    }

    pub fn forward_without_projection(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.clear_kv_cache();
        let input_embeds = self.language_model.embed_tokens().forward(input_ids)?;
        self.language_model
            .forward_embeds_without_projection(&input_embeds, None, 0)
    }
    pub fn setup_without_projection(
        &mut self,
        pixel_values: &Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        self.clear_kv_cache();
        let image_features = self
            .vision_tower
            .forward(pixel_values)?
            .apply(&self.multi_modal_projector)?;
        let image_features = crate::models::clip::div_l2_norm(&image_features)?;
        let text_features = self.language_model.embed_tokens().forward(input_ids)?;
        let input_embeds = Tensor::cat(&[image_features, text_features], 1)?;
        self.language_model
            .forward_embeds_without_projection(&input_embeds, None, 0)
    }
    pub fn clear_kv_cache(&mut self) {
        self.pos = 0;
        self.language_model.clear_kv_cache()
    }
}
