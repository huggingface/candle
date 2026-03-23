//! Config types for Qwen3-ASR.

pub mod asr_config;
pub mod audio_encoder_config;
pub mod text_config;

pub use asr_config::AsrConfig;
pub use audio_encoder_config::AudioEncoderConfig;
pub use text_config::{RopeScaling, TextConfig};
