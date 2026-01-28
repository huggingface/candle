//! Qwen3-TTS (text-to-speech) model.

pub mod config;
pub mod model;
pub mod speaker;
pub mod tokenizer_v2;

pub use config::{Qwen3TtsCodePredictorConfig, Qwen3TtsConfig, Qwen3TtsTalkerConfig};
pub use model::{GenerationParams, Qwen3Tts, VoiceClonePromptItem};
pub use tokenizer_v2::{Qwen3TtsTokenizerV2, Qwen3TtsTokenizerV2Config};
