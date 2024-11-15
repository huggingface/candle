//! Pixtral Language-Image Pre-Training
//!
//! Pixtral is an architecture trained for multimodal learning
//! using images paired with text descriptions.
//!
//! - Transformers Python [reference implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/pixtral)
//!

pub mod llava;
pub mod vision_model;

pub use llava::{Config, Model};
