//! CosyVoice3 Frontend Components
//!
//! Audio processing and tokenization for CosyVoice3.
//! - Mel spectrogram extraction
//! - Audio resampling
//! - Kaldi-compatible Fbank features

pub mod audio;

pub use audio::{kaldi_fbank, resample, KaldiFbank, MelSpectrogram};

