//! ACE-Step 1.5 music generation model.
//!
//! ACE-Step is a music generation foundation model based on a diffusion transformer (DiT)
//! architecture with text conditioning for lyrics and style prompts. Supports two
//! inference modes:
//!
//! - **DiT-only**: text encoder → DiT denoising → VAE decode (base/sft models)
//! - **LM+DiT**: Qwen3 LM generates metadata + audio codes → DiT conditioned on
//!   codes → VAE decode (turbo models, higher quality)
//!
//! ## Modules
//!
//! - [`pipeline`] — `AceStepPipeline` high-level API (text2music, cover)
//! - [`model`] — `AceStepConditionGenerationModel` (condition encoder + DiT + tokenizer)
//! - [`lm`] — `LmPipeline` for Qwen3-based audio code generation
//! - [`dit`] — `AceStepDiTModel` diffusion transformer
//! - [`condition`] — `ConditionEncoder` (text/lyric/timbre encoding)
//! - [`tokenizer`] — `AudioTokenizer` / `AudioTokenDetokenizer` / `ResidualFSQ`
//! - [`sampling`] — timestep schedules, Euler ODE/SDE, APG guidance
//! - [`vae`] — `AutoencoderOobleck` / `OobleckDecoder` for audio ↔ latent
//!
//! - [HuggingFace Models](https://huggingface.co/ACE-Step)
//! - [GitHub Repository](https://github.com/ACE-Step/ACE-Step-1.5)

pub mod condition;
pub mod dit;
pub mod lm;
pub mod model;
pub mod sampling;
pub mod tokenizer;
pub mod vae;

use candle_nn::Activation;

// -- AceStepConfig default value functions -----------------------------------

fn default_hidden_size() -> usize {
    2048
}
fn default_intermediate_size() -> usize {
    6144
}
fn default_num_hidden_layers() -> usize {
    24
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
fn default_hidden_act() -> Activation {
    Activation::Silu
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_use_sliding_window() -> bool {
    true
}
fn default_sliding_window() -> usize {
    128
}
fn default_in_channels() -> usize {
    192
}
fn default_patch_size() -> usize {
    2
}
fn default_text_hidden_dim() -> usize {
    1024
}
fn default_audio_acoustic_hidden_dim() -> usize {
    64
}
fn default_timbre_hidden_dim() -> usize {
    64
}
fn default_pool_window_size() -> usize {
    5
}
fn default_vocab_size() -> usize {
    64003
}
fn default_max_position_embeddings() -> usize {
    32768
}
fn default_fsq_dim() -> usize {
    2048
}
fn default_fsq_input_levels() -> Vec<usize> {
    vec![8, 8, 8, 5, 5, 5]
}
fn default_fsq_input_num_quantizers() -> usize {
    1
}
fn default_attention_bias() -> bool {
    false
}
fn default_attention_dropout() -> f64 {
    0.0
}
fn default_num_lyric_encoder_hidden_layers() -> usize {
    8
}
fn default_num_timbre_encoder_hidden_layers() -> usize {
    4
}
fn default_num_attention_pooler_hidden_layers() -> usize {
    2
}
fn default_num_audio_decoder_hidden_layers() -> usize {
    24
}
fn default_layer_types() -> Vec<String> {
    (0..24)
        .map(|i| {
            if i % 2 == 0 {
                "sliding_attention".to_string()
            } else {
                "full_attention".to_string()
            }
        })
        .collect()
}

/// Main configuration for the ACE-Step 1.5 DiT-based music generation model.
///
/// Deserializable from HuggingFace `config.json`. Supports both the base (3.5B)
/// and XL variants via optional encoder dimension overrides.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct AceStepConfig {
    // -- DiT dimensions ------------------------------------------------------
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

    // -- Encoder dimensions (XL only, optional) ------------------------------
    #[serde(default)]
    pub encoder_hidden_size: Option<usize>,
    #[serde(default)]
    pub encoder_intermediate_size: Option<usize>,
    #[serde(default)]
    pub encoder_num_attention_heads: Option<usize>,
    #[serde(default)]
    pub encoder_num_key_value_heads: Option<usize>,

    // -- Common parameters ---------------------------------------------------
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_use_sliding_window")]
    pub use_sliding_window: bool,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_text_hidden_dim")]
    pub text_hidden_dim: usize,
    #[serde(default = "default_audio_acoustic_hidden_dim")]
    pub audio_acoustic_hidden_dim: usize,
    #[serde(default = "default_timbre_hidden_dim")]
    pub timbre_hidden_dim: usize,
    #[serde(default = "default_pool_window_size")]
    pub pool_window_size: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_fsq_dim")]
    pub fsq_dim: usize,
    #[serde(default = "default_fsq_input_levels")]
    pub fsq_input_levels: Vec<usize>,
    #[serde(default = "default_fsq_input_num_quantizers")]
    pub fsq_input_num_quantizers: usize,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_attention_dropout")]
    pub attention_dropout: f64,

    // -- Sub-model layer counts ----------------------------------------------
    #[serde(default = "default_num_lyric_encoder_hidden_layers")]
    pub num_lyric_encoder_hidden_layers: usize,
    #[serde(default = "default_num_timbre_encoder_hidden_layers")]
    pub num_timbre_encoder_hidden_layers: usize,
    #[serde(default = "default_num_attention_pooler_hidden_layers")]
    pub num_attention_pooler_hidden_layers: usize,
    #[serde(default = "default_num_audio_decoder_hidden_layers")]
    pub num_audio_decoder_hidden_layers: usize,

    // -- Layer type schedule -------------------------------------------------
    #[serde(default = "default_layer_types")]
    pub layer_types: Vec<String>,

    // -- Turbo flag ----------------------------------------------------------
    #[serde(default)]
    pub is_turbo: bool,
}

impl AceStepConfig {
    /// Returns the encoder hidden size, falling back to the main `hidden_size`.
    pub fn encoder_hidden_size(&self) -> usize {
        self.encoder_hidden_size.unwrap_or(self.hidden_size)
    }

    /// Returns the encoder intermediate size, falling back to the main `intermediate_size`.
    pub fn encoder_intermediate_size(&self) -> usize {
        self.encoder_intermediate_size
            .unwrap_or(self.intermediate_size)
    }

    /// Returns the encoder attention head count, falling back to `num_attention_heads`.
    pub fn encoder_num_attention_heads(&self) -> usize {
        self.encoder_num_attention_heads
            .unwrap_or(self.num_attention_heads)
    }

    /// Returns the encoder key-value head count, falling back to `num_key_value_heads`.
    pub fn encoder_num_key_value_heads(&self) -> usize {
        self.encoder_num_key_value_heads
            .unwrap_or(self.num_key_value_heads)
    }
}

impl Default for AceStepConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            encoder_hidden_size: None,
            encoder_intermediate_size: None,
            encoder_num_attention_heads: None,
            encoder_num_key_value_heads: None,
            hidden_act: Activation::Silu,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            use_sliding_window: true,
            sliding_window: 128,
            in_channels: 192,
            patch_size: 2,
            text_hidden_dim: 1024,
            audio_acoustic_hidden_dim: 64,
            timbre_hidden_dim: 64,
            pool_window_size: 5,
            vocab_size: 64003,
            max_position_embeddings: 32768,
            fsq_dim: 2048,
            fsq_input_levels: vec![8, 8, 8, 5, 5, 5],
            fsq_input_num_quantizers: 1,
            attention_bias: false,
            attention_dropout: 0.0,
            num_lyric_encoder_hidden_layers: 8,
            num_timbre_encoder_hidden_layers: 4,
            num_attention_pooler_hidden_layers: 2,
            num_audio_decoder_hidden_layers: 24,
            layer_types: default_layer_types(),
            is_turbo: false,
        }
    }
}

// -- VaeConfig default value functions ---------------------------------------

fn default_vae_encoder_hidden_size() -> usize {
    128
}
fn default_vae_decoder_channels() -> usize {
    128
}
fn default_vae_decoder_input_channels() -> usize {
    64
}
fn default_vae_audio_channels() -> usize {
    2
}
fn default_vae_downsampling_ratios() -> Vec<usize> {
    vec![2, 4, 4, 6, 10]
}
fn default_vae_channel_multiples() -> Vec<usize> {
    vec![1, 2, 4, 8, 16]
}
fn default_vae_sampling_rate() -> usize {
    48000
}

/// Configuration for the ACE-Step VAE (variational autoencoder) that converts
/// between waveform audio and the latent space consumed by the DiT.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct VaeConfig {
    #[serde(default = "default_vae_encoder_hidden_size")]
    pub encoder_hidden_size: usize,
    #[serde(default = "default_vae_decoder_channels")]
    pub decoder_channels: usize,
    #[serde(default = "default_vae_decoder_input_channels")]
    pub decoder_input_channels: usize,
    #[serde(default = "default_vae_audio_channels")]
    pub audio_channels: usize,
    #[serde(default = "default_vae_downsampling_ratios")]
    pub downsampling_ratios: Vec<usize>,
    #[serde(default = "default_vae_channel_multiples")]
    pub channel_multiples: Vec<usize>,
    #[serde(default = "default_vae_sampling_rate")]
    pub sampling_rate: usize,
}

impl VaeConfig {
    /// Computes the total hop length as the product of all downsampling ratios.
    pub fn hop_length(&self) -> usize {
        self.downsampling_ratios.iter().product()
    }
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            encoder_hidden_size: 128,
            decoder_channels: 128,
            decoder_input_channels: 64,
            audio_channels: 2,
            downsampling_ratios: vec![2, 4, 4, 6, 10],
            channel_multiples: vec![1, 2, 4, 8, 16],
            sampling_rate: 48000,
        }
    }
}
