//! Automatic model loading from HuggingFace Hub.
//!
//! This module provides composable infrastructure for loading models dynamically,
//! similar to Python HuggingFace Transformers' `AutoModel.from_pretrained()`.
//!
//! ## Example
//!
//! ```no_run
//! use candle::{DType, Device};
//! use candle_transformers::auto::{AutoModelForCausalLM, AutoModelOptions};
//!
//! let model = AutoModelForCausalLM::from_pretrained(
//!     "Qwen/Qwen2-0.5B-Instruct",
//!     DType::F32,
//!     &Device::Cpu,
//!     AutoModelOptions::default(),
//! ).unwrap();
//! ```

mod config;
mod causal_lm;

pub use config::{AutoConfig, Weights};
pub use causal_lm::{Model, CausalLM, AutoModelForCausalLM, AutoModelOptions};
