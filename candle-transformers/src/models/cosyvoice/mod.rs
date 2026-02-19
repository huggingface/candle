//! CosyVoice3 Text-to-Speech Model
//!
//! CosyVoice3 is a multilingual zero-shot text-to-speech synthesizer that supports:
//! - SFT mode: Predefined speakers
//! - Zero-Shot: Voice cloning from reference audio
//! - Cross-Lingual: Cross-language voice cloning
//! - Instruct: Instruction-guided synthesis
//!
//! ## Architecture
//!
//! ```text
//! Input (text + prompt audio)
//!     │
//!     ▼
//! ┌─────────────────┐
//! │    Frontend     │ - Tokenizer, Mel extraction, Speaker embedding
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  CosyVoice3LM   │ - Qwen2-based autoregressive speech token generation
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  Flow Decoder   │ - DiT + Conditional Flow Matching
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  HiFT Vocoder   │ - Mel to waveform conversion
//! └────────┬────────┘
//!          │
//!          ▼
//! Output (24kHz waveform)
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use candle_transformers::models::cosyvoice::{
//!     CosyVoice3Config, CosyVoice3LM, CausalMaskedDiffWithDiT, CausalHiFTGenerator
//! };
//!
//! // Load configuration
//! let config = CosyVoice3Config::default();
//!
//! // Load model weights
//! let vb = VarBuilder::from_safetensors(&["model.safetensors"], DType::F16, device)?;
//!
//! // Create model components
//! let llm = CosyVoice3LM::new(&config.llm, vb.pp("llm"))?;
//! let vocoder = CausalHiFTGenerator::new(config.hift, vb.pp("hift"))?;
//!
//! // Run inference
//! let speech_tokens = llm.inference(&text_tokens, &prompt_tokens, ...)?;
//! let waveform = vocoder.inference(&mel, true)?;
//! ```
//!
//! ## References
//!
//! - [CosyVoice Paper](https://arxiv.org/abs/2407.05407)
//! - [GitHub Repository](https://github.com/FunAudioLLM/CosyVoice)

pub mod activations;
pub mod config;
pub mod flow;
pub mod frontend;
pub mod llm;
pub mod vocoder;

// Re-export main types from config
pub use config::{
    CFMConfig, CosyVoice3Config, CosyVoice3LMConfig, DiTConfig, F0PredictorConfig, FlowConfig,
    HiFTConfig, InferenceConfig, Qwen2Config, SamplingConfig,
};

// Re-export constants
pub use config::{
    DIT_DEPTH, DIT_DIM, DIT_HEADS, DIT_HEAD_DIM, HIFT_BASE_CHANNELS, ISTFT_HOP_LEN, ISTFT_N_FFT,
    LLM_DIM, LLM_VOCAB_SIZE, MEL_HOP_SIZE, MEL_N_FFT, MEL_N_MELS, SAMPLE_RATE, SPEECH_TOKEN_SIZE,
    SPK_EMBED_DIM, TOKEN_FRAME_RATE, TOKEN_MEL_RATIO,
};

// Re-export activation functions
pub use activations::{Elu, LeakyReLU, Mish, Snake, SnakeBeta, Swish, Tanh};

// Re-export flow components
pub use flow::dit::DiT;
pub use flow::embeddings::{
    AdaLayerNormZero, AdaLayerNormZeroFinal, InputEmbedding, RotaryEmbedding, TimestepEmbedding,
};
pub use flow::flow_matching::{CausalConditionalCFM, CausalMaskedDiffWithDiT};
pub use flow::pre_lookahead::PreLookaheadLayer;

// Re-export frontend components
pub use frontend::audio::{kaldi_fbank, resample, KaldiFbank, MelSpectrogram};
#[cfg(feature = "onnx")]
pub use frontend::onnx_models::CosyVoice3Frontend;

// Re-export LLM
pub use llm::cosyvoice3_lm::CosyVoice3LM;

// Re-export vocoder components
pub use vocoder::f0_predictor::{CausalConv1d, CausalConvRNNF0Predictor, CausalType};
pub use vocoder::hift_generator::{CausalHiFTGenerator, ResBlock};
pub use vocoder::istft::HiFTiSTFT;
pub use vocoder::source_module::{SineGen2, SourceModuleHnNSF};
pub use vocoder::stft::HiFTSTFT;
