//! Configuration for Granite-Docling (Idefics3) model.
//!
//! Architecture: SigLIP vision encoder → pixel shuffle connector → Llama-style causal decoder.
//!
//! References:
//! - [Model Card](https://huggingface.co/ibm-granite/granite-docling-258M)

use crate::models::siglip;

fn default_hidden_act() -> candle_nn::Activation {
    candle_nn::Activation::Silu
}

fn default_rope_theta() -> f64 {
    100_000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_max_position_embeddings() -> usize {
    8192
}

fn default_scale_factor() -> usize {
    4
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: candle_nn::Activation,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_head_dim() -> usize {
    64
}

impl TextConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Config {
    pub vision_config: siglip::VisionConfig,
    pub text_config: TextConfig,
    #[serde(default = "default_scale_factor")]
    pub scale_factor: usize,
    #[serde(default)]
    pub image_token_id: u32,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default)]
    pub eos_token_id: u32,
    #[serde(default)]
    pub pad_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub vocab_size: usize,
}

impl Config {
    /// Number of image tokens after pixel shuffle downsampling.
    /// = num_patches / scale_factor^2
    pub fn image_seq_len(&self) -> usize {
        let num_patches = self.vision_config.num_patches();
        num_patches / (self.scale_factor * self.scale_factor)
    }

    /// Dimension of vision features after pixel shuffle (before linear projection).
    /// = vision_hidden_size * scale_factor^2
    pub fn connector_input_dim(&self) -> usize {
        self.vision_config.hidden_size * self.scale_factor * self.scale_factor
    }
}
