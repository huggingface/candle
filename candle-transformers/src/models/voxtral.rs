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

use crate::models::llama::{Cache as LlamaCache, Config as LlamaConfig, Llama};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, Conv1d, Dropout, Embedding, LayerNorm, Linear,
    VarBuilder,
};

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
    pub dropout: f64,
    pub layer_dropout: f64,
    pub activation_dropout: f64,
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

impl VoxtralEncoderConfig {
    /// Dropout values are properly set sue to Whisper compatibility
    pub fn with_whisper_compatibility(mut self) -> Self {
        self.dropout = 0.0;
        self.layer_dropout = 0.0;
        self.activation_dropout = 0.0;
        self
    }
}

/// Custom cache for Voxtral
pub struct VoxtralCache {
    llama_cache: LlamaCache,
    audio_processed: bool,
    cached_audio_embeds: Option<Tensor>,
    cached_audio_positions: Option<Vec<(usize, usize)>>,
}

impl VoxtralCache {
    pub fn new(
        use_kv_cache: bool,
        dtype: Dtype,
        config: &LlamaConfig,
        device: &Device,
    ) -> Result<Self> {
        let llama_cache = LlamaCache::new(use_kv_cache, dtype, config, device)?;
        Ok(Self {
            llama_cache,
            audio_processed: false,
            cached_audio_embeds: None,
            cached_audio_positions: None,
        })
    }

    pub fn reset(&mut self) {
        self.llama_cache.reset();
        self.audio_processed = false;
        self.cached_audio_embeds = None;
        self.cached_audio_positions = None;
    }
}

/// Generate sinusodial position emdbeddings for audio sequence
fn sinusoids(num_positions: usize, embedding_dim: usize, device: &Device) -> Result<Tensor> {
    let half_dim = embedding_dim / 2;
    let mut emb = -(10000_f64.ln()) / (half_dim - 1) as f64;
    emb = (0..half_dim)
        .map(|i| (i as f64 * emb).exp())
        .collect::<Vec<_>>();
    emb = Tensor::new(emb.as_slice(), device)?;

    let pos = Tensor::arange(0, num_positions as i64, (DType::I64, device))?
        .to_dtype(DType::F64)?
        .unsqueeze(1)?;

    emb = emb.unsqueeze(0)?;
    let phase = pos.broadcast_mul(&emb)?;

    let sin = phase.sin()?;
    let cos = phase.cos()?;

    Tensor::cat(&[sin, cos], 1)
}

/// Safety clamp tensor values for different Dtypes
fn safe_clamp(x: &Tensor) -> Result<Tensor> {}
