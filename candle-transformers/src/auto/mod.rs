//! Automatic model loading from HuggingFace Hub.
//!
//! Similar to Python HuggingFace Transformers' `AutoModel.from_pretrained()`.
//!
//! ## Adding a new model
//!
//! 1. Ensure the model has `forward(&mut self, ids, offset)` + `clear_kv_cache(&mut self)`.
//! 2. Add `crate::impl_causal_lm!(MyModel, "my_model_type");` in the model file.
//! 3. Add a match arm in `auto/causal_lm.rs` `load_float_model()` or `load_gguf_model()`.

mod causal_lm;
mod config;

pub use causal_lm::{
    AutoModelForCausalLM, AutoModelOptions, CacheSnapshot, CausalLM, LayerKvSnapshot,
    QuantizationFormat,
};
pub use config::{AutoConfig, Weights};

/// GGUF model dispatch macro. Each model type must implement
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

/// Float model dispatch macro. Each entry deserialises `config.json` into
/// `$cfg_ty`, then calls the provided constructor closure.
#[macro_export]
macro_rules! make_auto_map {
    ($config:expr, $vb:expr, {
        $($name:literal => ($cfg_ty:ty, $ctor:expr)),+
        $(,)?
    }) => {{
        const SUPPORTED: &[&str] = &[$($name),+];
        match $config.model_type.to_lowercase().as_str() {
            $($name => {
                let cfg: $cfg_ty = $config.parse()?;
                let model = ($ctor)(cfg, $vb)?;
                Ok(Box::new(model) as Box<dyn $crate::auto::CausalLM>)
            },)+
            _ => Err(::candle::Error::Msg(format!(
                "unsupported model type '{}'. Supported float models: {}",
                $config.model_type,
                SUPPORTED.join(", "),
            ))),
        }
    }};
}

/// Implement [`CausalLM`] for a model struct that has
/// `forward(&mut self, ids, offset)` + `clear_kv_cache(&mut self)`.
#[macro_export]
macro_rules! impl_causal_lm {
    ($ty:ty, $name:literal) => {
        impl $crate::auto::CausalLM for $ty {
            fn model_type(&self) -> &'static str {
                $name
            }
            #[allow(unconditional_recursion)]
            fn forward(
                &mut self,
                ids: &::candle::Tensor,
                offset: usize,
            ) -> ::candle::Result<::candle::Tensor> {
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
