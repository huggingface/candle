use candle::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use super::vision_model;
use crate::models::mistral;

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub projector_hidden_act: candle_nn::Activation,
    pub text_config: mistral::Config,
    pub vision_config: vision_model::Config,
    pub image_token_index: usize,
    pub image_seq_length: usize,
}

#[derive(Debug, Clone)]
pub struct MultiModalProjector {
    linear_1: Linear,
    act: candle_nn::Activation,
    linear_2: Linear,
}

impl MultiModalProjector {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (hidden_v, hidden_t) = (cfg.vision_config.hidden_size, cfg.text_config.hidden_size);
        let linear_1 = linear(hidden_v, hidden_t, vb.pp("linear_1"))?;
        let linear_2 = linear(hidden_t, hidden_t, vb.pp("linear_2"))?;
        Ok(Self {
            linear_1,
            act: cfg.projector_hidden_act,
            linear_2,
        })
    }
}

impl Module for MultiModalProjector {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.linear_1)?
            .apply(&self.act)?
            .apply(&self.linear_2)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub multi_modal_projector: MultiModalProjector,
    pub language_model: mistral::Model,
    pub vision_tower: vision_model::Model,
    pub patch_size: usize,
    pub dtype: candle::DType,
    pub pos: usize,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let language_model = mistral::Model::new(&cfg.text_config, vb.pp("language_model"))?;
        let vision_tower = vision_model::Model::new(
            &cfg.vision_config,
            vb.pp("vision_tower").to_dtype(candle::DType::F32),
        )?;
        let multi_modal_projector = MultiModalProjector::new(
            cfg,
            vb.pp("multi_modal_projector").to_dtype(candle::DType::F32),
        )?;
        Ok(Self {
            multi_modal_projector,
            language_model,
            vision_tower,
            patch_size: cfg.vision_config.patch_size,
            dtype: vb.dtype(),
            pos: 0,
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
        self.pos = 0;
    }

    pub fn encode_image(&self, image: &Tensor) -> Result<Tensor> {
        let image_embeds = self.vision_tower.forward(image)?;
        self.multi_modal_projector.forward(&image_embeds)
    }

    pub fn lm_forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let logits = self.language_model.forward(input_ids, self.pos)?;
        self.pos += seq_len;
        Ok(logits)
    }

    pub fn lm_forward_embeds(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;
        let logits = self.language_model.forward_embeds(xs, None, self.pos)?;
        self.pos += seq_len;
        Ok(logits)
    }
}
