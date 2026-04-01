//! Automatic model loading from HuggingFace Hub.
//!
//! This module provides composable infrastructure for loading models dynamically,
//! similar to Python HuggingFace Transformers' `AutoModel.from_pretrained()`.
//!
//! ## Example
//!
//! ```ignore
//! use candle::Device;
//! use candle_transformers::auto::{AutoModelForCausalLM, AutoModelOptions};
//!
//! // Load from HF Hub (safetensors) — requires feature = "hf-hub"
//! # #[cfg(feature = "hf-hub")]
//! let model = AutoModelForCausalLM::from_pretrained(
//!     "Qwen/Qwen2-0.5B-Instruct",
//!     &Device::Cpu,
//!     AutoModelOptions::default(),
//! ).unwrap();
//!
//! // Load a local GGUF file
//! let model = AutoModelForCausalLM::from_gguf(
//!     std::path::Path::new("model.gguf"),
//!     &Device::Cpu,
//! ).unwrap();
//! ```
//!
//! ## Adding a new model
//!
//! 1. In the model file, call the appropriate macro variant:
//!    ```ignore
//!    // model has forward(input_ids, seqlen_offset) + clear_kv_cache():
//!    crate::impl_causal_lm!(MyModel, "my_model_type", with_reset);
//!    // model has forward(input_ids, seqlen_offset) but no clear_kv_cache():
//!    crate::impl_causal_lm!(MyModel, "my_model_type");
//!    // model has forward(xs) with no position offset:
//!    crate::impl_causal_lm!(MyModel, "my_model_type", stateless);
//!    ```
//! 2. Add a match arm in `auto/causal_lm.rs` `load_float_model()` or `load_gguf_model()`.

mod config;
mod causal_lm;

pub use config::{AutoConfig, Weights};
pub use causal_lm::{CausalLM, AutoModelForCausalLM, AutoModelOptions, QuantizationFormat};

/// Implement the [`CausalLM`] trait for a model struct.
///
/// # Variants
///
/// - `impl_causal_lm!(Type, "name")` — `forward(input_ids, seqlen_offset)`, no `clear_kv_cache`
/// - `impl_causal_lm!(Type, "name", with_reset)` — same + calls `Self::clear_kv_cache()`
/// - `impl_causal_lm!(Type, "name", stateless)` — `forward(xs)` with no offset (ignored)
/// - `impl_causal_lm!(Type, "name", stateless_with_reset)` — stateless + `clear_kv_cache`
#[macro_export]
macro_rules! impl_causal_lm {
    ($ty:ty, $name:literal) => {
        impl $crate::auto::CausalLM for $ty {
            fn model_type(&self) -> &'static str { $name }
            #[allow(unconditional_recursion)]
            fn forward(&mut self, ids: &::candle::Tensor, offset: usize) -> ::candle::Result<::candle::Tensor> {
                // Delegate to the model's own inherent method.
                // The `#[allow]` is needed because the compiler's recursion checker
                // cannot see through the macro that this calls the *inherent* forward,
                // not the CausalLM trait method.
                let m: &mut $ty = self;
                m.forward(ids, offset)
            }
        }
    };
    ($ty:ty, $name:literal, with_reset) => {
        impl $crate::auto::CausalLM for $ty {
            fn model_type(&self) -> &'static str { $name }
            #[allow(unconditional_recursion)]
            fn forward(&mut self, ids: &::candle::Tensor, offset: usize) -> ::candle::Result<::candle::Tensor> {
                let m: &mut $ty = self;
                m.forward(ids, offset)
            }
            #[allow(unconditional_recursion)]
            fn clear_kv_cache(&mut self) {
                let m: &mut $ty = self;
                m.clear_kv_cache()
            }
        }
    };
    ($ty:ty, $name:literal, stateless) => {
        impl $crate::auto::CausalLM for $ty {
            fn model_type(&self) -> &'static str { $name }
            #[allow(unconditional_recursion)]
            fn forward(&mut self, ids: &::candle::Tensor, _offset: usize) -> ::candle::Result<::candle::Tensor> {
                let m: &mut $ty = self;
                m.forward(ids)
            }
        }
    };
    ($ty:ty, $name:literal, stateless_with_reset) => {
        impl $crate::auto::CausalLM for $ty {
            fn model_type(&self) -> &'static str { $name }
            #[allow(unconditional_recursion)]
            fn forward(&mut self, ids: &::candle::Tensor, _offset: usize) -> ::candle::Result<::candle::Tensor> {
                let m: &mut $ty = self;
                m.forward(ids)
            }
            #[allow(unconditional_recursion)]
            fn clear_kv_cache(&mut self) {
                let m: &mut $ty = self;
                m.clear_kv_cache()
            }
        }
    };
}
