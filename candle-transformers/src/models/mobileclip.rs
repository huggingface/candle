//! Mobile CLIP model, combining a lightweight vision encoder with a text encoder
//!
//! A mobile-optimized CLIP implementation that uses:
//! - FastViT as the vision encoder
//! - OpenCLIP text encoder
//! - Projection layers to align the feature spaces
//!
//! See model details at:
//! - [FastViT](https://arxiv.org/abs/2303.14189)
//! - [OpenCLIP](https://github.com/mlfoundations/open_clip)
//!
//! References:
//! - [MobileVLM](https://huggingface.co/mobileVLM)
//! - [MetaCLIP](https://arxiv.org/abs/2309.16671)
//!

use super::fastvit;
use super::openclip::text_model;
use candle::{Result, Tensor, D};
use candle_nn::{Func, VarBuilder};

#[derive(Clone, Debug)]
pub struct MobileClipModel {
    text_model: text_model::OpenClipTextTransformer,
    vision_model: Func<'static>,
    text_projection: Tensor,
    logit_scale: Tensor,
}

#[derive(Clone, Debug)]
pub struct MobileClipConfig {
    pub text_config: text_model::Config,
    pub vision_config: fastvit::Config,
    pub image_size: usize,
}

impl MobileClipConfig {
    pub fn s1() -> Self {
        let text_config = text_model::Config::vit_base_patch32();
        let vision_config = fastvit::Config::mci1();
        Self {
            text_config,
            vision_config,
            image_size: 256,
        }
    }
    pub fn s2() -> Self {
        let text_config = text_model::Config::vit_base_patch32();
        let vision_config = fastvit::Config::mci2();
        Self {
            text_config,
            vision_config,
            image_size: 256,
        }
    }
}

impl MobileClipModel {
    pub fn new(vs: VarBuilder, c: &MobileClipConfig) -> Result<Self> {
        let vision_model = fastvit::fastvit(&c.vision_config, 512, vs.pp("visual.trunk"))?;
        let text_model = text_model::OpenClipTextTransformer::new(vs.pp("text"), &c.text_config)?;
        let text_projection = vs.get(
            (c.text_config.embed_dim, c.text_config.projection_dim),
            "text.text_projection",
        )?;
        let logit_scale = vs.get(&[], "logit_scale")?;
        Ok(Self {
            text_model,
            vision_model,
            text_projection,
            logit_scale,
        })
    }

    pub fn get_text_features(&self, input_ids: &Tensor) -> Result<Tensor> {
        input_ids
            .apply(&self.text_model)?
            .matmul(&self.text_projection)
    }

    pub fn get_image_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        pixel_values.apply(&self.vision_model)
    }

    pub fn forward(&self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let image_features = self.get_image_features(pixel_values)?;
        let text_features = self.get_text_features(input_ids)?;
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
