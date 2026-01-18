use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::de::DeserializeOwned;
use std::path::PathBuf;

use super::config::{AutoConfig, Weights};
use crate::models::{gemma, llama, mistral, phi3, qwen2};

pub trait Model: Send {
    fn model_type(&self) -> &'static str;
}

pub trait CausalLM: Model {
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>;

    fn clear_kv_cache(&mut self) {}
}

#[derive(Debug, Clone, Default)]
pub struct AutoModelOptions {
    pub revision: Option<String>,
    pub use_flash_attn: bool,
}

fn load<C, M, F>(config: &AutoConfig, build: F) -> Result<Box<dyn CausalLM>>
where
    C: DeserializeOwned,
    M: CausalLM + 'static,
    F: FnOnce(C) -> Result<M>,
{
    let cfg: C = config.parse()?;
    Ok(Box::new(build(cfg)?))
}

const SUPPORTED_MODELS: &[&str] = &["llama", "mistral", "phi3", "qwen2", "gemma"];

pub struct AutoModelForCausalLM;

impl AutoModelForCausalLM {
    #[cfg(feature = "hf-hub")]
    pub fn from_pretrained(
        model_id: &str,
        dtype: DType,
        device: &Device,
        options: AutoModelOptions,
    ) -> Result<Box<dyn CausalLM>> {
        let revision = options.revision.as_deref();
        let config = AutoConfig::from_pretrained(model_id, revision)?;
        let weights = Weights::from_pretrained(model_id, revision)?;
        let vb = unsafe { weights.into_var_builder(dtype, device)? };

        Self::load_model(&config, vb, dtype, device, options.use_flash_attn)
    }

    pub fn from_local(
        config_path: &std::path::Path,
        weight_paths: &[PathBuf],
        dtype: DType,
        device: &Device,
        use_flash_attn: bool,
    ) -> Result<Box<dyn CausalLM>> {
        let config = AutoConfig::from_local(config_path)?;
        let weights = Weights::from_local(weight_paths.to_vec());
        let vb = unsafe { weights.into_var_builder(dtype, device)? };

        Self::load_model(&config, vb, dtype, device, use_flash_attn)
    }

    fn load_model(
        config: &AutoConfig,
        vb: VarBuilder,
        dtype: DType,
        device: &Device,
        use_flash_attn: bool,
    ) -> Result<Box<dyn CausalLM>> {
        match config.model_type.to_lowercase().as_str() {
            "llama" => load(config, |cfg: llama::LlamaConfig| {
                let cfg = cfg.into_config(use_flash_attn);
                llama::LlamaForCausalLM::load(vb, &cfg, dtype, device)
            }),

            "mistral" => load(config, |mut cfg: mistral::Config| {
                cfg.use_flash_attn = use_flash_attn;
                mistral::Model::new(&cfg, vb)
            }),

            "phi3" => load(config, |cfg: phi3::Config| phi3::Model::new(&cfg, vb)),

            "qwen2" => load(config, |cfg: qwen2::Config| qwen2::ModelForCausalLM::new(&cfg, vb)),

            "gemma" => load(config, |cfg: gemma::Config| gemma::Model::new(use_flash_attn, &cfg, vb)),

            _ => Err(candle::Error::Msg(format!(
                "Unsupported model type: '{}'. Supported: {}",
                config.model_type,
                SUPPORTED_MODELS.join(", ")
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_models() {
        for model in SUPPORTED_MODELS {
            assert!(
                ["llama", "mistral", "phi3", "qwen2", "gemma"].contains(model),
                "Unknown model in SUPPORTED_MODELS: {model}"
            );
        }
    }
}
