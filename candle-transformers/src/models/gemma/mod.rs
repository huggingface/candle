//! Gemma model family implementations.
//!
//! This module contains implementations for Google's Gemma family of models:
//!
//! - [`gemma1`] - Original Gemma models (2B, 7B)
//! - [`gemma2`] - Gemma 2 models (2B, 9B, 27B)
//! - [`gemma3`] - Gemma 3 models (1B, 4B, 12B, 27B)
//! - [`quantized_gemma3`] - Quantized Gemma 3 (GGUF format)
//! - [`translate_gemma`] - TranslateGemma specialized translation models

pub mod gemma1;
pub mod gemma2;
pub mod gemma3;
pub mod quantized_gemma3;
pub mod translate_gemma;

// Backward compatibility - will be removed in a future version
#[deprecated(
    since = "0.9.0",
    note = "use `models::gemma::gemma1::{Config, Model}` instead"
)]
pub use gemma1::{Config, Model};
