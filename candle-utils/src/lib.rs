//! Helper utils for common Candle use cases.
//!
//! This crate provides convenient utilities for working with Candle.

pub mod device;
pub mod loader;
pub mod tensor;
pub mod tokenization;

// Re-exported for convenience.
pub use device::get_device;
pub use tensor::normalize_l2;
