//! Qwen3-ASR: Speech Recognition Model
//!
//! Multilingual automatic speech recognition (ASR) model supporting offline and
//! streaming transcription in 52+ languages and 22 Chinese dialects.
//!
//! Based on alan890104/qwen3-asr-rs (MIT).
//! Reference: <https://github.com/QwenLM/Qwen3-ASR>

pub mod audio_encoder;
pub mod kv_cache;
pub mod model;
mod rope;

use serde::Deserialize;

/// mRoPE configuration (Qwen3-ASR uses "default" scaling type internally).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub interleaved: bool,
    #[serde(default)]
    pub mrope_interleaved: bool,
}

/// Audio encoder configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncoderConfig {
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,
    #[serde(default = "default_encoder_layers")]
    pub encoder_layers: usize,
    #[serde(default = "default_encoder_attention_heads")]
    pub encoder_attention_heads: usize,
    #[serde(default = "default_encoder_ffn_dim")]
    pub encoder_ffn_dim: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default)]
    pub dropout: f64,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default = "default_activation_function")]
    pub activation_function: String,
    #[serde(default)]
    pub activation_dropout: f64,
    #[serde(default)]
    pub scale_embedding: bool,
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f64,
    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: usize,
    #[serde(default = "default_n_window")]
    pub n_window: usize,
    #[serde(default = "default_output_dim")]
    pub output_dim: usize,
    #[serde(default = "default_n_window_infer")]
    pub n_window_infer: usize,
    #[serde(default = "default_conv_chunksize")]
    pub conv_chunksize: usize,
    #[serde(default = "default_downsample_hidden_size")]
    pub downsample_hidden_size: usize,
}

/// Text model configuration (decoder-only transformer with mRoPE).
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
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
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_attention_dropout")]
    pub attention_dropout: f64,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
}

/// Top-level Qwen3-ASR configuration (matches HF Hub config.json).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub thinker_config: ThinkerConfig,
}

/// Thinker-level configuration (wraps audio + text model configs).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ThinkerConfig {
    #[serde(default)]
    pub audio_token_id: Option<u32>,
    #[serde(default)]
    pub audio_start_token_id: Option<u32>,
    #[serde(default)]
    pub audio_end_token_id: Option<u32>,
    #[serde(default)]
    pub audio_config: AudioEncoderConfig,
    #[serde(default)]
    pub text_config: TextConfig,
}

// Default values for audio encoder config.

fn default_num_mel_bins() -> usize {
    128
}
fn default_encoder_layers() -> usize {
    18
}
fn default_encoder_attention_heads() -> usize {
    14
}
fn default_encoder_ffn_dim() -> usize {
    3584
}
fn default_d_model() -> usize {
    896
}
fn default_activation_function() -> String {
    "gelu".to_string()
}
fn default_initializer_range() -> f64 {
    0.02
}
fn default_max_source_positions() -> usize {
    1500
}
fn default_n_window() -> usize {
    50
}
fn default_output_dim() -> usize {
    1024
}
fn default_n_window_infer() -> usize {
    800
}
fn default_conv_chunksize() -> usize {
    500
}
fn default_downsample_hidden_size() -> usize {
    480
}

// Default values for text config (Qwen3-ASR 0.6B).

fn default_vocab_size() -> usize {
    151_936
}
fn default_hidden_size() -> usize {
    1024
}
fn default_intermediate_size() -> usize {
    3072
}
fn default_num_hidden_layers() -> usize {
    28
}
fn default_num_attention_heads() -> usize {
    16
}
fn default_num_key_value_heads() -> usize {
    8
}
fn default_head_dim() -> usize {
    128
}
fn default_attention_dropout() -> f64 {
    0.0
}
fn default_attention_bias() -> bool {
    false
}
fn default_hidden_act() -> String {
    "silu".to_string()
}
fn default_max_position_embeddings() -> usize {
    65_536
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_use_cache() -> bool {
    true
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}

pub const DTYPE: candle::DType = candle::DType::F32;

impl Default for AudioEncoderConfig {
    fn default() -> Self {
        Self {
            num_mel_bins: default_num_mel_bins(),
            encoder_layers: default_encoder_layers(),
            encoder_attention_heads: default_encoder_attention_heads(),
            encoder_ffn_dim: default_encoder_ffn_dim(),
            d_model: default_d_model(),
            dropout: 0.0,
            attention_dropout: 0.0,
            activation_function: default_activation_function(),
            activation_dropout: 0.0,
            scale_embedding: false,
            initializer_range: default_initializer_range(),
            max_source_positions: default_max_source_positions(),
            n_window: default_n_window(),
            output_dim: default_output_dim(),
            n_window_infer: default_n_window_infer(),
            conv_chunksize: default_conv_chunksize(),
            downsample_hidden_size: default_downsample_hidden_size(),
        }
    }
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            head_dim: default_head_dim(),
            attention_dropout: 0.0,
            attention_bias: false,
            hidden_act: default_hidden_act(),
            max_position_embeddings: default_max_position_embeddings(),
            initializer_range: default_initializer_range(),
            rms_norm_eps: default_rms_norm_eps(),
            use_cache: true,
            tie_word_embeddings: false,
            rope_theta: default_rope_theta(),
            rope_scaling: None,
        }
    }
}
