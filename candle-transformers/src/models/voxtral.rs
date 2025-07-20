//! Voxtral implementation in Candle.
//!
//! Voxtral is a multi-modal model that combines:
//! - A Whisper-based audio encoder for processing audio features
//! - A multi-modal projector to map audio embeddings to text space
//! - A LLaMA language model for text generation
//!
//! Key characteristics:
//! - Audio processing through convolutional layers
//! - Sinusoidal position embeddings for audio
//! - Cross-modal attention between audio and text
//! - Autoregressive text generation conditioned on audio
//!

use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, linear_no_bias, Conv1d, LayerNorm, Linear};
use candle_transformers::llama::{Llama, LlamaConfig, LlamaForCausalLM};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct VoxtralEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub numm_attention_heads: usize,
    pub scale_embedding: bool,
    pub activation_function: String,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub initializer_range: f64,
    pub attention_dropout: f64,
    // According to transformers implementation,
    // Note: These are hardcoded to 0.0 for compatibility with Whisper modular architecture
    // TODO: Remove after Whisper refactor
    #[serde(default)]
    pub dropout: f64,
    #[serde(default)]
    pub layer_dropout: f64,
    #[serde(default)]
    activation_dropout: f64,
}

#[derive(Debug, Clone)]
pub struct VoxtralConfig {
    pub audio_config: VoxtralEncoderConfig,
    pub text_config: LlamaConfig,
    pub autio_token_id: usize,
    pub projector_hidden_act: String,
}

impl Default for VoxtralEncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 51866,
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            numm_attention_heads: 20,
            scale_embedding: false,
            activation_function: "gelu".to_string(),
            num_mel_bins: 128,
            max_source_positions: 1500,
            initializer_range: 0.02,
            attention_dropout: 0.0,
            // Hardcoded for Whisper compatibility
            dropout: 0.0,
            layer_dropout: 0.0,
            activation_dropout: 0.0,
        }
    }
}
