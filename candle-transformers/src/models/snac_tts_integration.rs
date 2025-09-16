//! SNAC Integration utilities for Text-to-Speech models
//!
//! This module provides convenient abstractions and utilities for integrating
//! SNAC (Multi-Scale Neural Audio Codec) with Text-to-Speech systems.
//! 
//! ## Usage Examples
//! 
//! ### Basic TTS Integration
//! ```rust
//! use candle_transformers::models::snac_tts_integration::*;
//! use candle_transformers::models::snac;
//! 
//! // Create SNAC codec for TTS
//! let config = snac::Config::default_tts();
//! let codec = SnacTtsCodec::new(&config, vb)?;
//! 
//! // Use in TTS pipeline
//! let audio_tokens = your_tts_model.generate_tokens(&text)?;
//! let audio_waveform = codec.tokens_to_audio(&audio_tokens)?;
//! ```
//! 
//! ### Qwen-based TTS with SNAC
//! ```rust
//! let tts_pipeline = QwenSnacTtsPipeline::new(qwen_model, snac_codec)?;
//! let audio = tts_pipeline.synthesize("Hello, world!", voice_prompt)?;
//! ```

use candle::{Result, Tensor, Device};
use candle_nn::VarBuilder;
use crate::models::snac::{self, Model as SnacModel};

/// A convenient wrapper around SNAC specifically optimized for TTS use cases
#[derive(Debug, Clone)]
pub struct SnacTtsCodec {
    model: SnacModel,
    device: Device,
}

impl SnacTtsCodec {
    /// Create a new SNAC TTS codec with the given configuration
    pub fn new(config: &snac::Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let model = SnacModel::new(config, vb)?;
        Ok(Self { model, device })
    }

    /// Create a SNAC TTS codec with default TTS settings (24kHz speech)
    pub fn new_default_tts(vb: VarBuilder) -> Result<Self> {
        let config = snac::Config::default_tts();
        Self::new(&config, vb)
    }

    /// Create a high-quality SNAC TTS codec for production use
    pub fn new_high_quality(vb: VarBuilder) -> Result<Self> {
        let config = snac::Config::high_quality_tts();
        Self::new(&config, vb)
    }

    /// Create a fast SNAC TTS codec for real-time applications
    pub fn new_fast(vb: VarBuilder) -> Result<Self> {
        let config = snac::Config::fast_tts();
        Self::new(&config, vb)
    }

    /// Convert audio tokens from a TTS model to waveform
    /// 
    /// Expected input shape: [batch_size, num_codebooks, sequence_length]
    /// Output shape: [batch_size, 1, audio_samples]
    pub fn tokens_to_audio(&self, tokens: &Tensor) -> Result<Tensor> {
        self.model.decode_from_tts_tokens(tokens)
    }

    /// Convert audio waveform to tokens for training TTS models
    /// 
    /// Input shape: [batch_size, 1, audio_samples] 
    /// Output shape: [batch_size, num_codebooks, sequence_length]
    pub fn audio_to_tokens(&self, audio: &Tensor) -> Result<Tensor> {
        self.model.encode_for_tts(audio)
    }

    /// Process a batch of audio tokens to waveforms efficiently
    pub fn batch_tokens_to_audio(&self, token_batches: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut results = Vec::with_capacity(token_batches.len());
        for tokens in token_batches {
            let audio = self.tokens_to_audio(tokens)?;
            results.push(audio);
        }
        Ok(results)
    }

    /// Get the number of codebooks (token streams) this codec uses
    pub fn num_codebooks(&self) -> usize {
        self.model.num_codebooks()
    }

    /// Get the sample rate of the codec
    pub fn sample_rate(&self) -> usize {
        self.model.get_sample_rate()
    }

    /// Convert duration in seconds to the expected number of tokens
    pub fn duration_to_tokens(&self, duration_seconds: f64) -> usize {
        let samples = (duration_seconds * self.sample_rate() as f64) as usize;
        self.model.samples_to_frames(samples)
    }

    /// Convert number of tokens to duration in seconds
    pub fn tokens_to_duration(&self, num_tokens: usize) -> f64 {
        let samples = self.model.frames_to_samples(num_tokens);
        samples as f64 / self.sample_rate() as f64
    }

    /// Pad token sequences to a target length (useful for batching)
    pub fn pad_tokens(&self, tokens: &Tensor, target_length: usize, pad_value: u32) -> Result<Tensor> {
        let (batch_size, num_codebooks, seq_len) = tokens.dims3()?;
        
        if seq_len >= target_length {
            return Ok(tokens.clone());
        }
        
        let pad_length = target_length - seq_len;
        let pad_shape = (batch_size, num_codebooks, pad_length);
        let pad_tensor = Tensor::full(pad_value, pad_shape, &self.device)?;
        
        Tensor::cat(&[tokens, &pad_tensor], 2)
    }

    /// Truncate token sequences to a maximum length
    pub fn truncate_tokens(&self, tokens: &Tensor, max_length: usize) -> Result<Tensor> {
        let (_, _, seq_len) = tokens.dims3()?;
        
        if seq_len <= max_length {
            return Ok(tokens.clone());
        }
        
        tokens.narrow(2, 0, max_length)
    }
}

/// Configuration for TTS models using SNAC
#[derive(Debug, Clone)]
pub struct TtsConfig {
    pub max_audio_length_seconds: f64,
    pub sample_rate: usize,
    pub pad_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            max_audio_length_seconds: 30.0,
            sample_rate: 24000,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }
}

/// Abstract trait for TTS models that can work with SNAC
pub trait SnacTtsModel {
    /// Generate audio tokens from text input
    fn generate_tokens(&mut self, text: &str, voice_prompt: Option<&Tensor>) -> Result<Tensor>;
    
    /// Get the model's vocabulary size for each codebook
    fn vocab_size(&self) -> usize;
    
    /// Clear any internal caches or state
    fn clear_cache(&mut self);
}

/// A complete TTS pipeline combining a language model with SNAC codec
#[derive(Debug)]
pub struct SnacTtsPipeline<T: SnacTtsModel> {
    tts_model: T,
    codec: SnacTtsCodec,
    _config: TtsConfig,
}

impl<T: SnacTtsModel> SnacTtsPipeline<T> {
    /// Create a new TTS pipeline
    pub fn new(tts_model: T, codec: SnacTtsCodec, config: Option<TtsConfig>) -> Self {
        let config = config.unwrap_or_default();
        Self {
            tts_model,
            codec,
            _config: config,
        }
    }

    /// Synthesize speech from text input
    pub fn synthesize(&mut self, text: &str, voice_prompt: Option<&Tensor>) -> Result<Tensor> {
        // Clear any previous state
        self.tts_model.clear_cache();
        
        // Generate audio tokens using the TTS model
        let tokens = self.tts_model.generate_tokens(text, voice_prompt)?;
        
        // Convert tokens to audio waveform using SNAC
        let audio = self.codec.tokens_to_audio(&tokens)?;
        
        Ok(audio)
    }

    /// Synthesize multiple texts in a batch (more efficient for many short texts)
    pub fn synthesize_batch(&mut self, texts: &[&str], voice_prompts: Option<&[Tensor]>) -> Result<Vec<Tensor>> {
        let mut results = Vec::with_capacity(texts.len());
        
        for (i, text) in texts.iter().enumerate() {
            let voice_prompt = voice_prompts.and_then(|prompts| prompts.get(i));
            let audio = self.synthesize(text, voice_prompt)?;
            results.push(audio);
        }
        
        Ok(results)
    }

    /// Get codec information
    pub fn codec_info(&self) -> CodecInfo {
        CodecInfo {
            sample_rate: self.codec.sample_rate(),
            num_codebooks: self.codec.num_codebooks(),
            compression_ratio: self.codec.model.config.get_compression_ratio(),
        }
    }
}

impl SnacTtsCodec {
    /// Get codec information
    pub fn codec_info(&self) -> CodecInfo {
        CodecInfo {
            sample_rate: self.sample_rate(),
            num_codebooks: self.num_codebooks(),
            compression_ratio: self.model.config.get_compression_ratio(),
        }
    }
}

/// Information about the audio codec
#[derive(Debug, Clone)]
pub struct CodecInfo {
    pub sample_rate: usize,
    pub num_codebooks: usize,
    pub compression_ratio: usize,
}

/// Utility functions for SNAC TTS integration
pub mod utils {
    use super::*;

    /// Create a voice embedding from a reference audio sample
    pub fn create_voice_embedding(codec: &SnacTtsCodec, reference_audio: &Tensor) -> Result<Tensor> {
        // Extract voice characteristics as tokens
        let tokens = codec.audio_to_tokens(reference_audio)?;
        
        // For voice cloning, typically we'd use the first few frames as the prompt
        let voice_frames = 50; // ~2 seconds at 24kHz with typical compression
        let (_batch_size, _num_codebooks, seq_len) = tokens.dims3()?;
        let prompt_len = voice_frames.min(seq_len);
        
        tokens.narrow(2, 0, prompt_len)
    }

    /// Validate that audio tokens have the expected format for SNAC
    pub fn validate_tokens(tokens: &Tensor, expected_codebooks: usize) -> Result<()> {
        let shape = tokens.shape();
        if shape.rank() != 3 {
            candle::bail!("Expected 3D tensor [batch, codebooks, sequence], got {:?}", shape);
        }
        
        let (_, codebooks, _) = tokens.dims3()?;
        if codebooks != expected_codebooks {
            candle::bail!(
                "Expected {} codebooks, got {}", 
                expected_codebooks, 
                codebooks
            );
        }
        
        Ok(())
    }

    /// Estimate the memory requirements for processing audio of given duration
    pub fn estimate_memory_usage(
        duration_seconds: f64,
        sample_rate: usize,
        num_codebooks: usize,
        batch_size: usize,
    ) -> MemoryEstimate {
        let samples = (duration_seconds * sample_rate as f64) as usize;
        let compression_ratio = 256; // Typical for SNAC
        let tokens = samples / compression_ratio;
        
        MemoryEstimate {
            audio_samples: batch_size * samples,
            token_count: batch_size * num_codebooks * tokens,
            estimated_bytes: batch_size * (samples * 4 + num_codebooks * tokens * 4), // 4 bytes per float/int
        }
    }

    #[derive(Debug, Clone)]
    pub struct MemoryEstimate {
        pub audio_samples: usize,
        pub token_count: usize,
        pub estimated_bytes: usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = snac::Config::default_tts();
        assert_eq!(config.sampling_rate, 24000);
        assert!(config.encoder_rates.len() > 0);
        
        let hq_config = snac::Config::high_quality_tts();
        assert!(hq_config.encoder_dim >= config.encoder_dim);
    }

    #[test]
    fn test_duration_calculations() {
        // This would require an actual SNAC model to test properly
        // In a real implementation, you'd load a model and test:
        // let codec = SnacTtsCodec::new_default_tts(vb)?;
        // assert_eq!(codec.duration_to_tokens(1.0), expected_tokens);
    }
}