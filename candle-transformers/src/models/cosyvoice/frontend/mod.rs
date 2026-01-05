/*
 * @Author: SpenserCai
 * @Date: 2026-01-04 10:00:56
 * @version:
 * @LastEditors: SpenserCai
 * @LastEditTime: 2026-01-05 11:19:04
 * @Description: file content
 */
//! CosyVoice3 Frontend Components
//!
//! Audio processing and tokenization for CosyVoice3.
//! - Mel spectrogram extraction
//! - Audio resampling
//! - Kaldi-compatible Fbank features
//! - ONNX-based speech tokenizer and speaker embedding extraction (requires `onnx` feature)

pub mod audio;
#[cfg(feature = "onnx")]
pub mod onnx_models;

pub use audio::{kaldi_fbank, resample, KaldiFbank, MelSpectrogram};
#[cfg(feature = "onnx")]
pub use onnx_models::CosyVoice3Frontend;
