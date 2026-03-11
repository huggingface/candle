use candle::{IndexOp, Tensor};
use candle_nn::{Activation, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use crate::models::lightonocr_2_1b::preprocessor::Preprocessor;
use crate::models::lightonocr_2_1b::projector::Projector;
use crate::models::pixtral::vision_model;
use crate::models::qwen3;
use candle::Result;

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    pub dtype: String,
    pub eos_token_id: usize,
    pub image_token_id: usize,
    pub pad_token_id: usize,
    pub projector_hidden_act: Activation,
    pub spatial_merge_size: usize,
    pub text_config: qwen3::Config,
    pub vision_config: vision_model::Config,
}

pub struct Model{
    pub vision_encoder: vision_model::Model,
    pub vision_config: vision_model::Config,
    pub projector: Projector,
    pub preprocessor: Preprocessor,
    pub language_model: qwen3::ModelForCausalLM,
    pub image_token_id: usize,
    pub model_config: Config
}

impl Model {
    pub fn new(cfg: Config, preprocessor: Preprocessor, vb: VarBuilder) -> Result<Self> {
        let model_vb = vb.pp("model");

        let vision_encoder = vision_model::Model::new(
            &cfg.vision_config, 
            model_vb.pp("vision_encoder"))?;

        let projector = Projector::new(
            cfg.vision_config.hidden_size , 
            model_vb.pp("vision_projection")
        )?;

        let language_model = qwen3::ModelForCausalLM::new(
            &cfg.text_config, 
            model_vb.pp("language_model")
        )?;

        Ok(Self { 
            vision_encoder, 
            vision_config: cfg.vision_config.clone(), 
            projector,
            preprocessor, 
            language_model, 
            image_token_id: cfg.image_token_id,
            model_config: cfg
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, pixel_values:&Tensor, offset: usize) -> Result<Tensor> {
        let image_features = self.vision_encoder.forward(pixel_values)?
        .squeeze(0)?;
        let (_, _, h, w) = pixel_values.dims4()?;
        let ph = h / self.vision_config.patch_size;
        let pw = w / self.vision_config.patch_size;

        let embeds = self.language_model.forward(input_ids, offset)?;

        let image_embeds = self.projector.forward(&image_features, ph, pw)?;
        let image_embeds = image_embeds.to_dtype(embeds.dtype())?;
        let embeds = self.splice_image_embeddings(input_ids, &embeds, &image_embeds)?;

        Ok(embeds)
    }

    pub fn splice_image_embeddings(&self, input_ids: &Tensor, embeds: &Tensor, image_embeds: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let ids: Vec<u32> = input_ids
            .squeeze(0)?
            .to_dtype(candle::DType::U32)?
            .to_vec1()?;

        let mut rows: Vec<Tensor> = Vec::with_capacity(seq_len);
        let mut img_idx = 0usize;

        for i in 0..seq_len {
            if ids[i] == self.image_token_id as u32 {
                let row = image_embeds.i(img_idx)?.unsqueeze(0)?;
                rows.push(row);
                img_idx += 1;
            } else {
                let row = embeds.i((0, i))?.unsqueeze(0)?;
                rows.push(row);
            }
        }
        Ok(Tensor::cat(&rows, 0)?
            .unsqueeze(0)?)  
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }

    
}