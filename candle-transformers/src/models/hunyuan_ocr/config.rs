//! Configuration for HunyuanOCR model.

use candle_nn::Activation;
use serde::Deserialize;

// ============================================================================
// Default value functions
// ============================================================================

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
fn default_vision_num_key_value_heads() -> usize {
    16
}
fn default_vision_num_channels() -> usize {
    3
}
fn default_vision_patch_size() -> usize {
    16
}
fn default_vision_max_image_size() -> usize {
    2048
}
fn default_vision_spatial_merge_size() -> usize {
    2
}
fn default_vision_out_hidden_size() -> usize {
    1024
}
fn default_vision_rms_norm_eps() -> f64 {
    1e-5
}
fn default_vision_hidden_act() -> Activation {
    Activation::Gelu
}
fn default_interpolate_mode() -> String {
    "bilinear".to_string()
}
fn default_max_vit_seq_len() -> usize {
    16384
}

fn default_text_vocab_size() -> usize {
    120818
}
fn default_text_hidden_size() -> usize {
    1024
}
fn default_text_intermediate_size() -> usize {
    3584
}
fn default_text_num_hidden_layers() -> usize {
    24
}
fn default_text_num_attention_heads() -> usize {
    16
}
fn default_text_num_key_value_heads() -> usize {
    8
}
fn default_text_max_position_embeddings() -> usize {
    32768
}
fn default_text_rms_norm_eps() -> f64 {
    1e-5
}
fn default_text_rope_theta() -> f64 {
    10000.0
}
fn default_text_hidden_act() -> Activation {
    Activation::Silu
}

fn default_xdrope_section() -> Option<Vec<usize>> {
    Some(vec![16, 16, 16, 16])
}

fn default_im_start_id() -> u32 {
    120118
}
fn default_im_end_id() -> u32 {
    120119
}
fn default_image_token_id() -> u32 {
    120120
}
fn default_im_newline_id() -> u32 {
    120121
}

// ============================================================================
// Vision Configuration
// ============================================================================

/// Vision encoder configuration.
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
    #[serde(default = "default_vision_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_vision_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_max_image_size")]
    pub max_image_size: usize,
    #[serde(default = "default_vision_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_vision_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_vision_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_interpolate_mode")]
    pub interpolate_mode: String,
    #[serde(default = "default_max_vit_seq_len")]
    pub max_vit_seq_len: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_vision_hidden_size(),
            intermediate_size: default_vision_intermediate_size(),
            num_hidden_layers: default_vision_num_hidden_layers(),
            num_attention_heads: default_vision_num_attention_heads(),
            num_key_value_heads: default_vision_num_key_value_heads(),
            num_channels: default_vision_num_channels(),
            patch_size: default_vision_patch_size(),
            max_image_size: default_vision_max_image_size(),
            spatial_merge_size: default_vision_spatial_merge_size(),
            out_hidden_size: default_vision_out_hidden_size(),
            rms_norm_eps: default_vision_rms_norm_eps(),
            hidden_act: default_vision_hidden_act(),
            interpolate_mode: default_interpolate_mode(),
            max_vit_seq_len: default_max_vit_seq_len(),
        }
    }
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ============================================================================
// RoPE Scaling Configuration
// ============================================================================

/// RoPE scaling configuration for xDRoPE.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(rename = "type")]
    pub scaling_type: Option<String>,
    pub factor: Option<f64>,
    pub alpha: Option<f64>,
    #[serde(default = "default_xdrope_section")]
    pub xdrope_section: Option<Vec<usize>>,
}

// ============================================================================
// Text Configuration
// ============================================================================

/// Text decoder configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    #[serde(default = "default_text_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_text_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_text_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_text_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_text_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_text_num_key_value_heads")]
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    #[serde(default = "default_text_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_text_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_text_rope_theta")]
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_text_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_text_vocab_size(),
            hidden_size: default_text_hidden_size(),
            intermediate_size: default_text_intermediate_size(),
            num_hidden_layers: default_text_num_hidden_layers(),
            num_attention_heads: default_text_num_attention_heads(),
            num_key_value_heads: default_text_num_key_value_heads(),
            head_dim: Some(128),
            max_position_embeddings: default_text_max_position_embeddings(),
            rms_norm_eps: default_text_rms_norm_eps(),
            rope_theta: default_text_rope_theta(),
            rope_scaling: None,
            hidden_act: default_text_hidden_act(),
            attention_bias: false,
            attention_dropout: 0.0,
            bos_token_id: Some(120000),
            eos_token_id: Some(120020),
        }
    }
}

impl TextConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

// ============================================================================
// Combined Configuration
// ============================================================================

/// Complete HunyuanOCR model configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vision_config: VisionConfig,
    // Text config fields at root level (HuggingFace format)
    #[serde(flatten)]
    pub text_config: TextConfig,

    // Special token IDs
    #[serde(default = "default_im_start_id", alias = "image_start_token_id")]
    pub im_start_id: u32,
    #[serde(default = "default_im_end_id", alias = "image_end_token_id")]
    pub im_end_id: u32,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    #[serde(default = "default_im_newline_id", alias = "image_newline_token_id")]
    pub im_newline_id: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vision_config: VisionConfig::default(),
            text_config: TextConfig::default(),
            im_start_id: default_im_start_id(),
            im_end_id: default_im_end_id(),
            image_token_id: default_image_token_id(),
            im_newline_id: default_im_newline_id(),
        }
    }
}
