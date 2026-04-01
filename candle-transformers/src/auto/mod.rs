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

/// Build the `load_gguf_model` dispatch from a compact GGUF model registry.
///
/// Every listed model type must implement
/// `from_gguf(ct: Content, reader: &mut R, device: &Device) -> Result<Self>`.
#[macro_export]
macro_rules! make_gguf_map {
    ($arch:expr, $content:expr, $reader:expr, $device:expr, {
        $($name:literal => $model_ty:ty),+
        $(,)?
    }) => {{
        const SUPPORTED: &[&str] = &[$($name),+];
        match $arch.to_lowercase().as_str() {
            $(
                $name => Ok(Box::new(
                    <$model_ty>::from_gguf($content, $reader, $device)?
                ) as Box<dyn $crate::auto::CausalLM>),
            )+
            _ => Err(::candle::Error::Msg(format!(
                "unsupported GGUF architecture '{}'. Supported: {}",
                $arch,
                SUPPORTED.join(", "),
            ))),
        }
    }};
}

/// Build the `load_float_model` dispatch table from a compact model registry.
///
/// # Usage
///
/// ```ignore
/// make_auto_map!(config, vb, use_flash_attn, {
///     "mistral" => (mistral::Config, mistral::Model),
///     "llama"   => (llama::LlamaConfig, llama::LlamaForCausalLM),
/// })
/// ```
///
/// Each entry deserialises `config.json` into `$cfg_ty`, sets
/// `cfg.use_flash_attn = $flash`, then calls `<$model_ty>::new(&cfg, vb)`.
/// All model configs must expose a `pub use_flash_attn: bool` field
/// (add `#[serde(default)]` to keep it backward-compatible with existing
/// `config.json` files that omit the field).
#[macro_export]
macro_rules! make_auto_map {
    ($config:expr, $vb:expr, $flash:expr, {
        $($name:literal => ($cfg_ty:ty, $model_ty:ty)),+
        $(,)?
    }) => {{
        const SUPPORTED: &[&str] = &[$($name),+];
        match $config.model_type.to_lowercase().as_str() {
            $($name => {
                let mut cfg: $cfg_ty = $config.parse()?;
                cfg.use_flash_attn = $flash;
                Ok(Box::new(<$model_ty>::new(&cfg, $vb)?) as Box<dyn $crate::auto::CausalLM>)
            },)+
            _ => Err(::candle::Error::Msg(format!(
                "unsupported model type '{}'. Supported float models: {}",
                $config.model_type,
                SUPPORTED.join(", "),
            ))),
        }
    }};
}

/// Implement the [`CausalLM`] trait for a model struct.
///
/// The model must already have:
/// - `fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>`
/// - `fn clear_kv_cache(&mut self)`
///
/// Every causal LM must be stateful (hold a KV cache), so `clear_kv_cache` is
/// always required — there is no variant that omits it.
///
/// The `#[allow(unconditional_recursion)]` annotations silence a false-positive
/// from the compiler's recursion checker, which cannot see through the typed
/// binding that this calls the *inherent* method, not the trait method.
///
/// # Usage
/// ```ignore
/// crate::impl_causal_lm!(Model, "mistral");
/// ```
#[macro_export]
macro_rules! impl_causal_lm {
    ($ty:ty, $name:literal) => {
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
}

/// Generate a `*ForCausalLM` wrapper struct that bundles an inner model with
/// its KV cache — the standard pattern for models whose cache is external
/// (e.g. LLaMA, Granite).
///
/// Generates `new(cfg, vb)`, `forward`, `clear_kv_cache`, and the
/// [`CausalLM`] trait impl.  The inner model must have:
/// - `InnerModel::load(vb: VarBuilder, cfg: &InnerCfg) -> Result<Self>`
/// - `InnerCache::new(use_kv_cache: bool, dtype: DType, cfg: &InnerCfg, device: &Device) -> Result<Self>`
/// - `inner_model.forward(ids, offset, &mut cache) -> Result<Tensor>`
///
/// # Usage
/// ```ignore
/// crate::causal_lm_wrapper!(
///     LlamaForCausalLM, "llama",
///     model: llama::Llama,
///     cache: llama::Cache,
///     cfg:   llama::LlamaConfig,
///     inner_cfg: llama::Config,          // optional: config type accepted by load/Cache::new
/// );
/// ```
#[macro_export]
macro_rules! causal_lm_wrapper {
    (
        $wrapper:ident, $name:literal,
        model: $model_ty:ty,
        cache: $cache_ty:ty,
        cfg:   $cfg_ty:ty,
    ) => {
        pub struct $wrapper {
            model: $model_ty,
            cache: $cache_ty,
        }

        impl $wrapper {
            pub fn new(cfg: &$cfg_ty, vb: ::candle_nn::VarBuilder) -> ::candle::Result<Self> {
                let model = <$model_ty>::load(vb.clone(), cfg)?;
                let cache = <$cache_ty>::new(true, vb.dtype(), cfg, vb.device())?;
                Ok(Self { model, cache })
            }

            pub fn forward(
                &mut self,
                input_ids: &::candle::Tensor,
                seqlen_offset: usize,
            ) -> ::candle::Result<::candle::Tensor> {
                self.model.forward(input_ids, seqlen_offset, &mut self.cache)
            }

            pub fn clear_kv_cache(&mut self) {
                self.cache.kvs.iter_mut().for_each(|kv| *kv = None);
            }
        }

        $crate::impl_causal_lm!($wrapper, $name);
    };
}
