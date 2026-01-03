//! PaddleOCR-VL configuration structures.
//!
//! Defines the configuration for the vision encoder, text decoder, and combined model.

use candle_nn::Activation;
use serde::Deserialize;

fn default_vision_hidden_size() -> usize {
    1152
}

fn default_vision_intermediate_size() -> usize {
    4304
}

fn default_vision_num_hidden_layers() -> usize {
    27
}

fn default_vision_num_attention_heads() -> usize {
    16
}

fn default_vision_num_channels() -> usize {
    3
}

fn default_vision_image_size() -> usize {
    384
}

fn default_vision_patch_size() -> usize {
    14
}

fn default_vision_hidden_act() -> Activation {
    Activation::GeluPytorchTanh
}

fn default_vision_layer_norm_eps() -> f64 {
    1e-6
}

fn default_vision_attention_dropout() -> f64 {
    0.0
}

fn default_vision_spatial_merge_size() -> usize {
    2
}

/// Vision encoder configuration for PaddleOCR-VL.
///
/// Uses a NaViT-style dynamic resolution visual encoder with 2D rotary position embeddings.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,

    #[serde(default = "default_vision_num_hidden_layers")]
    pub num_hidden_layers: usize,

    #[serde(default = "default_vision_num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default = "default_vision_num_channels")]
    pub num_channels: usize,

    #[serde(default = "default_vision_image_size")]
    pub image_size: usize,

    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,

    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,

    #[serde(default = "default_vision_layer_norm_eps")]
    pub layer_norm_eps: f64,

    #[serde(default = "default_vision_attention_dropout")]
    pub attention_dropout: f64,

    #[serde(default = "default_vision_spatial_merge_size")]
    pub spatial_merge_size: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_vision_hidden_size(),
            intermediate_size: default_vision_intermediate_size(),
            num_hidden_layers: default_vision_num_hidden_layers(),
            num_attention_heads: default_vision_num_attention_heads(),
            num_channels: default_vision_num_channels(),
            image_size: default_vision_image_size(),
            patch_size: default_vision_patch_size(),
            hidden_act: default_vision_hidden_act(),
            layer_norm_eps: default_vision_layer_norm_eps(),
            attention_dropout: default_vision_attention_dropout(),
            spatial_merge_size: default_vision_spatial_merge_size(),
        }
    }
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

fn default_vocab_size() -> usize {
    103424
}

fn default_hidden_size() -> usize {
    1024
}

fn default_intermediate_size() -> usize {
    3072
}

fn default_num_hidden_layers() -> usize {
    18
}

fn default_num_attention_heads() -> usize {
    16
}

fn default_num_key_value_heads() -> usize {
    2
}

fn default_hidden_act() -> Activation {
    Activation::Silu
}

fn default_max_position_embeddings() -> usize {
    131072
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f64 {
    500000.0
}

fn default_head_dim() -> usize {
    128
}

fn default_use_bias() -> bool {
    false
}

fn default_tie_word_embeddings() -> bool {
    false
}

fn default_image_token_id() -> u32 {
    100295
}

fn default_video_token_id() -> u32 {
    101307
}

fn default_vision_start_token_id() -> u32 {
    101305
}

fn default_vision_end_token_id() -> u32 {
    101306
}

fn default_tokens_per_second() -> usize {
    25
}

/// RoPE scaling configuration for multimodal position embeddings.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    /// Sections for multimodal RoPE: [temporal, height, width].
    /// Splits head_dim/2 into 3 parts for 3D position encoding.
    /// Default: [16, 24, 24] (total = 64 = head_dim/2 for head_dim=128)
    #[serde(default = "default_mrope_section")]
    pub mrope_section: Vec<usize>,

    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_mrope_section() -> Vec<usize> {
    vec![16, 24, 24]
}

impl Default for RopeScaling {
    fn default() -> Self {
        Self {
            mrope_section: default_mrope_section(),
            rope_type: Some("default".to_string()),
        }
    }
}

/// Combined configuration for PaddleOCR-VL model.
///
/// The text model parameters are at the top level (not nested in `text_config`),
/// following the HuggingFace format where the main model config contains LLM params directly.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    // Vision config (nested)
    #[serde(default)]
    pub vision_config: VisionConfig,

    // Text model parameters (at top level)
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,

    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,

    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps", alias = "rms_norm_eps")]
    pub layer_norm_eps: f64,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    #[serde(default = "default_head_dim")]
    pub head_dim: usize,

    #[serde(default = "default_use_bias")]
    pub use_bias: bool,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    // Special token IDs
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,

    #[serde(default = "default_video_token_id")]
    pub video_token_id: u32,

    #[serde(default = "default_vision_start_token_id")]
    pub vision_start_token_id: u32,

    #[serde(default = "default_vision_end_token_id")]
    pub vision_end_token_id: u32,

    /// RoPE scaling configuration for multimodal position embeddings.
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    /// Tokens per second for video temporal position encoding.
    #[serde(default = "default_tokens_per_second")]
    pub tokens_per_second: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vision_config: VisionConfig::default(),
            vocab_size: default_vocab_size(),
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            hidden_act: default_hidden_act(),
            max_position_embeddings: default_max_position_embeddings(),
            layer_norm_eps: default_rms_norm_eps(),
            rope_theta: default_rope_theta(),
            head_dim: default_head_dim(),
            use_bias: default_use_bias(),
            tie_word_embeddings: default_tie_word_embeddings(),
            image_token_id: default_image_token_id(),
            video_token_id: default_video_token_id(),
            vision_start_token_id: default_vision_start_token_id(),
            vision_end_token_id: default_vision_end_token_id(),
            rope_scaling: Some(RopeScaling::default()),
            tokens_per_second: default_tokens_per_second(),
        }
    }
}

/// Helper struct for text config (used internally).
/// This provides a view of the text-related config fields.
#[derive(Debug, Clone)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub head_dim: usize,
    pub use_bias: bool,
    pub tie_word_embeddings: bool,
    /// Multimodal RoPE sections: [temporal, height, width].
    pub mrope_section: Vec<usize>,
}

impl From<&Config> for TextConfig {
    fn from(cfg: &Config) -> Self {
        let mrope_section = cfg
            .rope_scaling
            .as_ref()
            .map(|rs| rs.mrope_section.clone())
            .unwrap_or_else(default_mrope_section);
        Self {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            hidden_act: cfg.hidden_act,
            max_position_embeddings: cfg.max_position_embeddings,
            rms_norm_eps: cfg.layer_norm_eps,
            rope_theta: cfg.rope_theta,
            head_dim: cfg.head_dim,
            use_bias: cfg.use_bias,
            tie_word_embeddings: cfg.tie_word_embeddings,
            mrope_section,
        }
    }
}
