#![allow(unused)]
use crate::models::{gemma, siglip};
use candle::{Module, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Config {
    pub vision_config: siglip::VisionConfig,
    pub text_config: gemma::Config,
    pub ignore_index: i64,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub image_token_index: usize,
    pub pad_token_id: usize,
    pub projection_dim: usize,
    pub vocab_size: usize,
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
        let vision_tower =
            siglip::VisionModel::new(&cfg.vision_config, false, vb.pp("vision_tower"))?;
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

    pub fn clear_kv_cache(&mut self) {
        self.pos = 0;
        self.language_model.clear_kv_cache()
    }
}
