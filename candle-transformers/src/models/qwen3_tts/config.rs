//! Qwen3-TTS configuration structs.

use std::collections::HashMap;

use candle_nn::Activation;
use serde::Deserialize;

fn default_mrope_section() -> Vec<usize> {
    vec![16, 24, 24]
}

fn default_hidden_act() -> Activation {
    Activation::Silu
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    10000.0
}

fn default_max_position_embeddings() -> usize {
    32768
}

fn default_num_code_groups() -> usize {
    32
}

fn default_text_hidden_size() -> usize {
    2048
}

fn default_text_vocab_size() -> usize {
    151_936
}

fn default_mel_dim() -> usize {
    128
}

fn default_speaker_enc_dim() -> usize {
    1024
}

fn default_speaker_enc_channels() -> Vec<usize> {
    vec![512, 512, 512, 512, 1536]
}

fn default_speaker_enc_kernel_sizes() -> Vec<usize> {
    vec![5, 3, 3, 3, 1]
}

fn default_speaker_enc_dilations() -> Vec<usize> {
    vec![1, 2, 3, 4, 1]
}

fn default_speaker_enc_attention_channels() -> usize {
    128
}

fn default_speaker_enc_res2net_scale() -> usize {
    8
}

fn default_speaker_enc_se_channels() -> usize {
    128
}

fn default_speaker_enc_sample_rate() -> usize {
    24_000
}

fn default_codec_pad_id() -> i64 {
    4196
}

fn default_codec_bos_id() -> i64 {
    4197
}

fn default_codec_eos_id() -> i64 {
    4198
}

fn default_codec_think_id() -> i64 {
    4202
}

fn default_codec_nothink_id() -> i64 {
    4203
}

fn default_codec_think_bos_id() -> i64 {
    4204
}

fn default_codec_think_eos_id() -> i64 {
    4205
}

fn default_tts_pad_token_id() -> i64 {
    151_671
}

fn default_tts_bos_token_id() -> i64 {
    151_672
}

fn default_tts_eos_token_id() -> i64 {
    151_673
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default = "default_mrope_section")]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub interleaved: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsTalkerConfig {
    #[serde(default)]
    pub code_predictor_config: Qwen3TtsCodePredictorConfig,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default = "default_num_code_groups")]
    pub num_code_groups: usize,
    #[serde(default = "default_text_hidden_size")]
    pub text_hidden_size: usize,
    #[serde(default = "default_text_vocab_size")]
    pub text_vocab_size: usize,
    #[serde(default)]
    pub pad_token_id: Option<i64>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_codec_eos_id")]
    pub codec_eos_token_id: i64,
    #[serde(default = "default_codec_think_id")]
    pub codec_think_id: i64,
    #[serde(default)]
    pub codec_language_id: HashMap<String, i64>,
    #[serde(default = "default_codec_nothink_id")]
    pub codec_nothink_id: i64,
    #[serde(default = "default_codec_think_bos_id")]
    pub codec_think_bos_id: i64,
    #[serde(default = "default_codec_think_eos_id")]
    pub codec_think_eos_id: i64,
    #[serde(default = "default_codec_pad_id")]
    pub codec_pad_id: i64,
    #[serde(default = "default_codec_bos_id")]
    pub codec_bos_id: i64,
    #[serde(default)]
    pub spk_id: HashMap<String, Vec<i64>>,
    #[serde(default)]
    pub spk_is_dialect: HashMap<String, serde_json::Value>,
}

impl Qwen3TtsTalkerConfig {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn sliding_window(&self) -> Option<usize> {
        if self.use_sliding_window {
            self.sliding_window
        } else {
            None
        }
    }

    pub fn mrope_section(&self) -> Vec<usize> {
        self.rope_scaling
            .as_ref()
            .map(|s| s.mrope_section.clone())
            .unwrap_or_else(default_mrope_section)
    }

    pub fn mrope_interleaved(&self) -> bool {
        self.rope_scaling
            .as_ref()
            .map(|s| s.interleaved)
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsCodePredictorConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub max_window_layers: Option<usize>,
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    #[serde(default = "default_num_code_groups")]
    pub num_code_groups: usize,
}

impl Default for Qwen3TtsCodePredictorConfig {
    fn default() -> Self {
        Self {
            vocab_size: 2048,
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 5,
            num_attention_heads: 16,
            num_key_value_heads: Some(8),
            head_dim: Some(128),
            hidden_act: default_hidden_act(),
            max_position_embeddings: default_max_position_embeddings(),
            rope_theta: default_rope_theta(),
            rms_norm_eps: default_rms_norm_eps(),
            attention_bias: false,
            attention_dropout: 0.0,
            use_sliding_window: false,
            sliding_window: Some(4096),
            max_window_layers: Some(28),
            layer_types: None,
            num_code_groups: default_num_code_groups(),
        }
    }
}

impl Qwen3TtsCodePredictorConfig {
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn layer_types(&self) -> Vec<LayerType> {
        if let Some(layer_types) = &self.layer_types {
            return layer_types.clone();
        }
        let max_window_layers = self.max_window_layers.unwrap_or(self.num_hidden_layers);
        let sliding_window = if self.use_sliding_window {
            self.sliding_window
        } else {
            None
        };
        (0..self.num_hidden_layers)
            .map(|i| {
                if sliding_window.is_some() && i >= max_window_layers {
                    LayerType::SlidingAttention
                } else {
                    LayerType::FullAttention
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsSpeakerEncoderConfig {
    #[serde(default = "default_mel_dim")]
    pub mel_dim: usize,
    #[serde(default = "default_speaker_enc_dim")]
    pub enc_dim: usize,
    #[serde(default = "default_speaker_enc_channels")]
    pub enc_channels: Vec<usize>,
    #[serde(default = "default_speaker_enc_kernel_sizes")]
    pub enc_kernel_sizes: Vec<usize>,
    #[serde(default = "default_speaker_enc_dilations")]
    pub enc_dilations: Vec<usize>,
    #[serde(default = "default_speaker_enc_attention_channels")]
    pub enc_attention_channels: usize,
    #[serde(default = "default_speaker_enc_res2net_scale")]
    pub enc_res2net_scale: usize,
    #[serde(default = "default_speaker_enc_se_channels")]
    pub enc_se_channels: usize,
    #[serde(default = "default_speaker_enc_sample_rate")]
    pub sample_rate: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsConfig {
    pub talker_config: Qwen3TtsTalkerConfig,
    #[serde(default)]
    pub speaker_encoder_config: Option<Qwen3TtsSpeakerEncoderConfig>,
    #[serde(default)]
    pub tokenizer_type: Option<String>,
    #[serde(default)]
    pub tts_model_size: Option<String>,
    #[serde(default)]
    pub tts_model_type: Option<String>,
    #[serde(default)]
    pub im_start_token_id: Option<i64>,
    #[serde(default)]
    pub im_end_token_id: Option<i64>,
    #[serde(default = "default_tts_pad_token_id")]
    pub tts_pad_token_id: i64,
    #[serde(default = "default_tts_bos_token_id")]
    pub tts_bos_token_id: i64,
    #[serde(default = "default_tts_eos_token_id")]
    pub tts_eos_token_id: i64,
}
