use candle::{
    Context, CpuStorage, CustomOp1, DType, Error, Layout, Module, Result, Shape, Tensor, WithDType,
    D,
};
use candle_nn::VarBuilder;
use config::Gemma3Config;
use mmproj::Gemma3MultiModalProjector;

pub mod config;
mod mmproj;

use crate::models::siglip;

use super::gemma3;

struct NonZero {}

impl NonZero {
    // Sequential version
    fn nonzero<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Vec<u32> {
        let n = layout.dims().len();
        let mut result = Vec::new();
        let mut indices = vec![0u32; n];
        for (i, v) in vs.iter().enumerate() {
            if !v.is_zero() {
                let mut idx = i;
                for (dim_index, dim) in layout.dims().iter().enumerate().rev() {
                    let d = idx % dim;
                    indices[dim_index] = u32::try_from(d).unwrap();
                    idx /= dim;
                }
                result.extend_from_slice(&indices);
            }
        }
        result
    }
}

impl CustomOp1 for NonZero {
    fn name(&self) -> &'static str {
        "nonzero"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            return Err(Error::RequiresContiguous { op: "nonzero" });
        }
        let result = match storage {
            candle::CpuStorage::U8(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::U32(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::I64(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::BF16(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::F16(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::F32(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::F64(vs) => self.nonzero(vs, layout),
        };
        let index_len = layout.dims().len();
        let result_len = result.len() / index_len;
        let result = CpuStorage::U32(result);
        let shape = Shape::from_dims(&[result_len, index_len]);
        Ok((result, shape))
    }
}

pub trait NonZeroOp {
    fn nonzero(&self) -> Result<Tensor>;
}

impl NonZeroOp for Tensor {
    fn nonzero(&self) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(candle::Error::RequiresContiguous { op: "nonzero" });
        }
        let original_device = self.device();
        self.to_device(&candle::Device::Cpu)?
            .apply_op1_no_bwd(&NonZero {})?
            .to_device(original_device)
    }
}

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

    fn forward(
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
