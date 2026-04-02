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

impl TextConfig {
    /// Build TextConfig from GGUF metadata keys (llama.* namespace).
    pub fn from_gguf(
        metadata: &std::collections::HashMap<String, candle::quantized::gguf_file::Value>,
    ) -> candle::Result<Self> {
        let get = |key: &str| {
            metadata
                .get(key)
                .ok_or_else(|| candle::Error::Msg(format!("missing GGUF metadata: {key}")))
        };
        Ok(Self {
            vocab_size: get("llama.vocab_size")?.to_u32()? as usize,
            hidden_size: get("llama.embedding_length")?.to_u32()? as usize,
            intermediate_size: get("llama.feed_forward_length")?.to_u32()? as usize,
            num_hidden_layers: get("llama.block_count")?.to_u32()? as usize,
            num_attention_heads: get("llama.attention.head_count")?.to_u32()? as usize,
            num_key_value_heads: get("llama.attention.head_count_kv")?.to_u32()? as usize,
            head_dim: get("llama.attention.key_length")?.to_u32()? as usize,
            hidden_act: candle_nn::Activation::Silu,
            max_position_embeddings: get("llama.context_length")?.to_u32()? as usize,
            rms_norm_eps: get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64,
            rope_theta: get("llama.rope.freq_base")?.to_f32()? as f64,
            attention_bias: false,
            mlp_bias: false,
            tie_word_embeddings: true,
        })
    }
}

/// Vision-related config extracted from mmproj GGUF metadata.
#[derive(Clone, Debug)]
pub struct QuantizedVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub layer_norm_eps: f64,
    pub projection_dim: usize,
    pub scale_factor: usize,
}

impl QuantizedVisionConfig {
    /// Build from mmproj GGUF metadata keys (clip.vision.* namespace).
    pub fn from_gguf(
        metadata: &std::collections::HashMap<String, candle::quantized::gguf_file::Value>,
    ) -> candle::Result<Self> {
        let get = |key: &str| {
            metadata
                .get(key)
                .ok_or_else(|| candle::Error::Msg(format!("missing GGUF metadata: {key}")))
        };
        Ok(Self {
            hidden_size: get("clip.vision.embedding_length")?.to_u32()? as usize,
            intermediate_size: get("clip.vision.feed_forward_length")?.to_u32()? as usize,
            num_hidden_layers: get("clip.vision.block_count")?.to_u32()? as usize,
            num_attention_heads: get("clip.vision.attention.head_count")?.to_u32()? as usize,
            image_size: get("clip.vision.image_size")?.to_u32()? as usize,
            patch_size: get("clip.vision.patch_size")?.to_u32()? as usize,
            layer_norm_eps: get("clip.vision.attention.layer_norm_epsilon")?.to_f32()? as f64,
            projection_dim: get("clip.vision.projection_dim")?.to_u32()? as usize,
            scale_factor: get("clip.vision.projector.scale_factor")?.to_u32()? as usize,
        })
    }

    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn image_seq_len(&self) -> usize {
        self.num_patches() / (self.scale_factor * self.scale_factor)
    }

    pub fn connector_input_dim(&self) -> usize {
        self.hidden_size * self.scale_factor * self.scale_factor
    }
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
