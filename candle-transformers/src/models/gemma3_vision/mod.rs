use candle::{Context, DType, Module, Result, Tensor, D};
use candle_nn::VarBuilder;
use config::Gemma3Config;
use mmproj::Gemma3MultiModalProjector;

pub mod config;
mod mmproj;

use crate::models::siglip;

use super::{deepseek2::NonZeroOp, gemma3};

pub struct Gemma3Model {
    language_model: gemma3::Model,
    multi_modal_projector: Option<Gemma3MultiModalProjector>,
    vision_tower: Option<siglip::VisionModel>,
    cfg: Gemma3Config,
    dtype: DType,
}

impl Gemma3Model {
    pub fn new(use_flash_attn: bool, cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        match cfg {
            Gemma3Config::Text(text_cfg) => Ok(Self {
                dtype: vb.dtype(),
                language_model: gemma3::Model::new(use_flash_attn, text_cfg, vb)?,
                multi_modal_projector: None,
                vision_tower: None,
                cfg: cfg.clone(),
            }),
            Gemma3Config::WithVision {
                text_config,
                vision_config,
                image_token_index,
                mm_tokens_per_image: _,
            } => {
                assert!(*image_token_index < text_config.vocab_size);
                Ok(Self {
                    multi_modal_projector: Some(Gemma3MultiModalProjector::new(
                        cfg,
                        vb.pp("multi_modal_projector"),
                    )?),
                    vision_tower: Some(siglip::VisionModel::new(
                        vision_config,
                        false,
                        vb.pp("vision_tower").pp("vision_model"),
                    )?),
                    language_model: gemma3::Model::new(
                        use_flash_attn,
                        text_config,
                        vb.pp("language_model"),
                    )?,
                    cfg: cfg.clone(),
                    dtype: vb.dtype(),
                })
            }
        }
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut input_embeds = self.language_model.embed_tokens(input_ids)?;
        if let Some(pixel_values) = pixel_values {
            let vision_tower = self
                .vision_tower
                .as_ref()
                .context("This model does not support vision.")?;
            let multi_modal_projector = self.multi_modal_projector.as_ref().unwrap();
            let Gemma3Config::WithVision {
                image_token_index, ..
            } = &self.cfg
            else {
                unreachable!()
            };

            let vision_outputs = vision_tower.forward(&pixel_values.to_dtype(self.dtype)?)?;
            let image_features = multi_modal_projector.forward(&vision_outputs)?;

            let special_image_mask = input_ids
                .eq(*image_token_index as f64)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .to_dtype(DType::U32)?;

            let mask_flat = special_image_mask.flatten_all()?;
            let mut x_flat = input_embeds.flatten_all()?;
            let src_flat = image_features.flatten_all()?;

            let indices = mask_flat.nonzero()?.squeeze(1)?;
            let current_vals = x_flat.gather(&indices, 0)?;
            let diff = (src_flat - current_vals)?;
            x_flat = x_flat.scatter_add(&indices, &diff, 0)?;

            input_embeds = x_flat.reshape(input_embeds.shape())?;
        };
        self.language_model
            .forward_embeds(input_ids, input_embeds, seqlen_offset)
    }
}
