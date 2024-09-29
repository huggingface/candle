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
            language_model,
            vision_tower,
            multi_modal_projector,
        })
    }

    pub fn forward(&mut self, _input_ids: &Tensor, _pos: usize) -> Result<Tensor> {
        todo!()
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache()
    }
}
