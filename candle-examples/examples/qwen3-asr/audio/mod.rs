//! Audio utilities: input decoding, normalization, resampling, and chunking.

pub mod chunking;
pub mod decode;
pub mod input;
pub mod normalize;
pub mod resample;

pub use input::AudioInput;

/// Target ASR sample rate (matches the official Python implementation).
pub const SAMPLE_RATE_HZ: u32 = 16_000;
