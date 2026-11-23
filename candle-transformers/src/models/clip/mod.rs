//! Contrastive Language-Image Pre-Training
//!
//! Contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
//! pairs of images with related texts.
//!
//! - 💻 [GH Link](https://github.com/openai/CLIP)
//! - 💻 Transformers Python [reference implementation](https://github.com/huggingface/transformers/tree/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip)
//! - 🤗 [HF Model](https://huggingface.co/openai/clip-vit-large-patch14-336)
//!

use self::{
    text_model::{Activation, ClipTextTransformer},
    vision_model::ClipVisionTransformer,
};
use candle::{Result, Tensor, D};

pub mod text_model;
pub mod vision_model;

#[derive(Clone, Debug)]
pub struct ClipModel {
    text_model: ClipTextTransformer,
    vision_model: ClipVisionTransformer,
    visual_projection: candle_nn::Linear,
    text_projection: candle_nn::Linear,
    logit_scale: Tensor,
}

#[derive(Clone, Debug)]
pub enum EncoderConfig {
    Text(text_model::ClipTextConfig),
    Vision(vision_model::ClipVisionConfig),
}

impl EncoderConfig {
    pub fn embed_dim(&self) -> usize {
        match self {
            Self::Text(c) => c.embed_dim,
            Self::Vision(c) => c.embed_dim,
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
            Self::Text(_c) => Activation::QuickGelu,
            Self::Vision(c) => c.activation,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClipConfig {
    pub text_config: text_model::ClipTextConfig,
    pub vision_config: vision_model::ClipVisionConfig,
    pub logit_scale_init_value: f32,
    pub image_size: usize,
}

/// Mirror of the HF `config.json` layout. HF keeps `image_size` inside
/// `vision_config`; the struct-level field is filled from there so a
/// default can never silently disagree with the vision weights.
#[derive(serde::Deserialize)]
struct ClipConfigHf {
    text_config: text_model::ClipTextConfig,
    vision_config: vision_model::ClipVisionConfig,
    #[serde(default = "default_logit_scale")]
    logit_scale_init_value: f32,
}

fn default_logit_scale() -> f32 {
    2.6592
}

impl<'de> serde::Deserialize<'de> for ClipConfig {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let hf = ClipConfigHf::deserialize(d)?;
        let image_size = hf.vision_config.image_size;
        Ok(Self {
            text_config: hf.text_config,
            vision_config: hf.vision_config,
            logit_scale_init_value: hf.logit_scale_init_value,
            image_size,
        })
    }
}

impl ClipConfig {
    // base image size is 224, model size is 600Mb
    pub fn vit_base_patch32() -> Self {
        let text_config = text_model::ClipTextConfig::vit_base_patch32();
        let vision_config = vision_model::ClipVisionConfig::vit_base_patch32();

        Self {
            text_config,
            vision_config,
            logit_scale_init_value: 2.6592,
            image_size: 224,
        }
    }
}

impl ClipModel {
    pub fn new(vs: candle_nn::VarBuilder, c: &ClipConfig) -> Result<Self> {
        let text_model = ClipTextTransformer::new(vs.pp("text_model"), &c.text_config)?;
        let vision_model = ClipVisionTransformer::new(vs.pp("vision_model"), &c.vision_config)?;
        let visual_projection = candle_nn::linear_no_bias(
            c.vision_config.embed_dim,
            c.vision_config.projection_dim,
            vs.pp("visual_projection"),
        )?;
        let text_projection = candle_nn::linear_no_bias(
            c.text_config.embed_dim,
            c.text_config.projection_dim,
            vs.pp("text_projection"),
        )?;
        // originally nn.Parameter
        let logit_scale = if vs.contains_tensor("logit_scale") {
            vs.get(&[], "logit_scale")?
        } else {
            Tensor::new(&[c.logit_scale_init_value], vs.device())?
        };
        Ok(Self {
            text_model,
            vision_model,
            visual_projection,
            text_projection,
            logit_scale,
        })
    }

    pub fn get_text_features(&self, input_ids: &Tensor) -> Result<Tensor> {
        input_ids
            .apply(&self.text_model)?
            .apply(&self.text_projection)
    }

    pub fn get_image_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        pixel_values
            .apply(&self.vision_model)?
            .apply(&self.visual_projection)
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

#[cfg(test)]
mod tests {
    use super::*;

    const B32: &str = r#"{
        "logit_scale_init_value": 2.6592,
        "text_config": {
            "hidden_act": "quick_gelu",
            "hidden_size": 512,
            "intermediate_size": 2048,
            "max_position_embeddings": 77,
            "num_attention_heads": 8,
            "num_hidden_layers": 12,
            "projection_dim": 512,
            "vocab_size": 49408
        },
        "vision_config": {
            "hidden_act": "quick_gelu",
            "hidden_size": 768,
            "image_size": 224,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_channels": 3,
            "num_hidden_layers": 12,
            "patch_size": 32,
            "projection_dim": 512
        }
    }"#;

    #[test]
    fn clip_config_parses_from_hf_json() {
        let config: ClipConfig = serde_json::from_str(B32).unwrap();
        assert_eq!(config.text_config.embed_dim, 512);
        assert_eq!(config.text_config.vocab_size, 49408);
        assert_eq!(config.vision_config.embed_dim, 768);
        assert_eq!(config.vision_config.patch_size, 32);
        assert_eq!(config.logit_scale_init_value, 2.6592);
        // image_size mirrors vision_config, never a struct-level default.
        assert_eq!(config.image_size, 224);
        assert_eq!(
            config.text_config.activation,
            text_model::Activation::QuickGelu
        );
        // Both presets must stay identical to the parsed HF shape.
        let preset = ClipConfig::vit_base_patch32();
        assert_eq!(preset.image_size, config.image_size);
        assert_eq!(preset.text_config.embed_dim, config.text_config.embed_dim);
        assert_eq!(
            preset.vision_config.embed_dim,
            config.vision_config.embed_dim
        );
    }

    #[test]
    fn clip_config_image_size_follows_vision_config() {
        let large: ClipConfig = serde_json::from_str(
            &B32.replace("224", "336")
                .replace("\"patch_size\": 32", "\"patch_size\": 14"),
        )
        .unwrap();
        assert_eq!(large.image_size, 336);
        assert_eq!(large.vision_config.patch_size, 14);
    }

    #[test]
    fn clip_config_uses_defaults_where_hf_omits_fields() {
        let trimmed = r#"{
            "text_config": {
                "hidden_size": 512, "intermediate_size": 2048,
                "max_position_embeddings": 77, "num_attention_heads": 8,
                "num_hidden_layers": 12, "projection_dim": 512, "vocab_size": 49408
            },
            "vision_config": {
                "hidden_size": 768, "image_size": 224, "intermediate_size": 3072,
                "num_attention_heads": 12, "num_hidden_layers": 12,
                "patch_size": 32, "projection_dim": 512
            }
        }"#;
        let config: ClipConfig = serde_json::from_str(trimmed).unwrap();
        assert_eq!(config.logit_scale_init_value, 2.6592);
        assert_eq!(config.vision_config.num_channels, 3);
        assert_eq!(
            config.vision_config.activation,
            text_model::Activation::QuickGelu
        );
    }

    #[test]
    fn clip_config_rejects_unknown_activation() {
        let bad = B32.replace("quick_gelu", "not_an_activation");
        assert!(serde_json::from_str::<ClipConfig>(&bad).is_err());
    }

    #[test]
    fn clip_config_accepts_both_quick_gelu_spellings() {
        // HF checkpoints carry both the current `quick_gelu` and the
        // legacy `quickgelu` spelling of the CLIP activation.
        for spelling in ["quick_gelu", "quickgelu"] {
            let json = B32.replace("quick_gelu", spelling);
            let config: ClipConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(
                config.text_config.activation,
                text_model::Activation::QuickGelu
            );
        }
    }
}
