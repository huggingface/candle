use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::de::DeserializeOwned;
use std::path::{Path, PathBuf};

use super::config::{AutoConfig, Weights};
use crate::models::{
    chatglm, deepseek2, gemma, gemma2, gemma3, granite, helium, llama, mistral, mixtral, olmo,
    olmo2, phi, phi3, qwen2, qwen2_moe, qwen3, qwen3_moe, stable_lm, starcoder2, yi,
};
use crate::models::{
    quantized_gemma3, quantized_llama, quantized_phi, quantized_phi3, quantized_qwen2,
    quantized_qwen3,
};

// ---------------------------------------------------------------------------
// CausalLM trait
// ---------------------------------------------------------------------------

/// A causal language model that can run autoregressive inference.
///
/// Implementors return logits of shape `[batch, vocab_size]` for the **last**
/// input token from [`forward`].  Callers (sampling loops, beam search, etc.)
/// rely on this contract.
pub trait CausalLM: Send {
    /// The HuggingFace `model_type` string, e.g. `"llama"` or `"mistral"`.
    fn model_type(&self) -> &'static str;

    /// Run one forward step.
    ///
    /// - `input_ids`: token IDs, shape `[batch, seq_len]`.
    /// - `seqlen_offset`: number of tokens **already in the KV cache** (0 on
    ///   the first call, grows by `seq_len` on each subsequent call).
    ///
    /// Returns logits `[batch, vocab_size]` for the last input token only.
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>;

    /// Reset the internal KV cache.  No-op by default; override for models
    /// with stateful attention caches.
    fn clear_kv_cache(&mut self) {}
}

// ---------------------------------------------------------------------------
// QuantizationFormat + AutoModelOptions
// ---------------------------------------------------------------------------

/// Weight format for model loading.
#[derive(Debug, Clone)]
pub enum QuantizationFormat {
    /// GGUF quantized weights — routes to the `quantized_*` model variants.
    Gguf {
        /// Path to the local `.gguf` file.
        path: PathBuf,
    },
}

/// Options for [`AutoModelForCausalLM`] loaders.
#[derive(Debug, Clone, Default)]
pub struct AutoModelOptions {
    /// HF Hub revision / branch / commit hash.  Defaults to `"main"`.
    pub revision: Option<String>,
    /// Override the dtype used for safetensors loading.
    /// Defaults to `BF16` on CUDA/Metal and `F32` on CPU.
    pub dtype: Option<DType>,
    /// Enable flash-attention kernels where supported.
    /// Requires the `flash-attn` feature; silently ignored on CPU.
    pub use_flash_attn: bool,
    /// When set, load GGUF weights instead of safetensors.
    /// `from_pretrained` will use this path; `from_gguf` ignores it.
    pub quantization: Option<QuantizationFormat>,
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

fn load<C, M, F>(config: &AutoConfig, build: F) -> Result<Box<dyn CausalLM>>
where
    C: DeserializeOwned,
    M: CausalLM + 'static,
    F: FnOnce(C) -> Result<M>,
{
    let cfg: C = config.parse()?;
    Ok(Box::new(build(cfg)?))
}

fn default_dtype(device: &Device) -> DType {
    if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    }
}

// ---------------------------------------------------------------------------
// Supported model lists (kept as documentation; derived from match arms below)
// ---------------------------------------------------------------------------

/// Float (safetensors) model types supported by [`AutoModelForCausalLM`].
pub const SUPPORTED_FLOAT_MODELS: &[&str] = &[
    // Llama family
    "llama",
    // Mistral / Mixtral
    "mistral",
    "mixtral",
    // Phi family
    "phi",
    "phi3",
    // Qwen family
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
    // Gemma family
    "gemma",
    "gemma2",
    "gemma3",
    // Other decoder-only models
    "starcoder2",
    "deepseek_v2",
    "olmo",
    "olmo2",
    "stablelm",
    "yi",
    "helium",
    "granite",
    "chatglm",
];

/// GGUF model types supported by [`AutoModelForCausalLM::from_gguf`].
pub const SUPPORTED_GGUF_MODELS: &[&str] = &[
    "llama", "phi", "phi3", "qwen2", "qwen3", "gemma", "gemma3",
];

// ---------------------------------------------------------------------------
// AutoModelForCausalLM
// ---------------------------------------------------------------------------

/// Factory for loading causal language models.
pub struct AutoModelForCausalLM;

impl AutoModelForCausalLM {
    /// Load a model from the HuggingFace Hub.
    ///
    /// Requires the `hf-hub` feature.  Automatically selects the right
    /// architecture from `config.json`.  Pass `options.quantization = Some(…)`
    /// to load GGUF weights instead of safetensors.
    #[cfg(feature = "hf-hub")]
    pub fn from_pretrained(
        model_id: &str,
        device: &Device,
        options: AutoModelOptions,
    ) -> Result<Box<dyn CausalLM>> {
        let revision = options.revision.as_deref();
        let config = AutoConfig::from_pretrained(model_id, revision)?;
        let dtype = options.dtype.unwrap_or_else(|| default_dtype(device));

        if let Some(QuantizationFormat::Gguf { path }) = &options.quantization {
            return Self::load_gguf(path, &config.model_type, device);
        }

        let weights = Weights::from_pretrained(model_id, revision)?;
        let vb = unsafe { weights.into_var_builder(dtype, device)? };
        Self::load_float_model(&config, vb, dtype, device, options.use_flash_attn)
    }

    /// Load a model from local files.
    pub fn from_local(
        config_path: &Path,
        weight_paths: &[PathBuf],
        device: &Device,
        options: AutoModelOptions,
    ) -> Result<Box<dyn CausalLM>> {
        let config = AutoConfig::from_local(config_path)?;
        let dtype = options.dtype.unwrap_or_else(|| default_dtype(device));

        if let Some(QuantizationFormat::Gguf { path }) = &options.quantization {
            return Self::load_gguf(path, &config.model_type, device);
        }

        let weights = Weights::from_local(weight_paths.to_vec());
        let vb = unsafe { weights.into_var_builder(dtype, device)? };
        Self::load_float_model(&config, vb, dtype, device, options.use_flash_attn)
    }

    /// Load directly from a local `.gguf` file.
    ///
    /// The model architecture is detected from the `general.architecture`
    /// field embedded in the GGUF metadata.
    pub fn from_gguf(path: &Path, device: &Device) -> Result<Box<dyn CausalLM>> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| candle::Error::Msg(format!("cannot open GGUF file: {e}")))?;
        let content = candle::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| candle::Error::Msg(format!("failed to read GGUF: {e}")))?;
        // Extract arch string before moving `content` into the loader.
        // The string is cloned/converted to owned so the borrow on `content`
        // is released before `content` is moved.
        // to_string() returns Result<&String> — clone to own before moving content.
        let arch = content
            .metadata
            .get("general.architecture")
            .ok_or_else(|| candle::Error::Msg("GGUF missing general.architecture".to_string()))?
            .to_string()
            .map_err(|_| candle::Error::Msg("general.architecture is not a string".to_string()))?
            .clone();

        Self::load_gguf_from_content(content, &mut file, &arch, device)
    }

    // ------------------------------------------------------------------
    // Internal: float (safetensors) model dispatch
    // ------------------------------------------------------------------

    fn load_float_model(
        config: &AutoConfig,
        vb: VarBuilder,
        dtype: DType,
        device: &Device,
        use_flash_attn: bool,
    ) -> Result<Box<dyn CausalLM>> {
        match config.model_type.to_lowercase().as_str() {
            // ── Llama ──────────────────────────────────────────────────────
            "llama" => load(config, |cfg: llama::LlamaConfig| {
                let cfg = cfg.into_config(use_flash_attn);
                llama::LlamaForCausalLM::load(vb, &cfg, dtype, device)
            }),

            // ── Mistral / Mixtral ──────────────────────────────────────────
            "mistral" => load(config, |mut cfg: mistral::Config| {
                cfg.use_flash_attn = use_flash_attn;
                mistral::Model::new(&cfg, vb)
            }),

            "mixtral" => load(config, |mut cfg: mixtral::Config| {
                cfg.use_flash_attn = use_flash_attn;
                mixtral::Model::new(&cfg, vb)
            }),

            // ── Phi family ─────────────────────────────────────────────────
            "phi" | "phi-msft" => {
                load(config, |cfg: phi::Config| phi::Model::new(&cfg, vb))
            }

            "phi3" => load(config, |cfg: phi3::Config| phi3::Model::new(&cfg, vb)),

            // ── Qwen family ────────────────────────────────────────────────
            "qwen2" => load(config, |cfg: qwen2::Config| {
                qwen2::ModelForCausalLM::new(&cfg, vb)
            }),

            "qwen2_moe" => load(config, |cfg: qwen2_moe::Config| {
                qwen2_moe::Model::new(&cfg, vb)
            }),

            "qwen3" => load(config, |cfg: qwen3::Config| {
                qwen3::ModelForCausalLM::new(&cfg, vb)
            }),

            "qwen3_moe" => load(config, |cfg: qwen3_moe::Config| {
                qwen3_moe::ModelForCausalLM::new(&cfg, vb)
            }),

            // ── Gemma family ───────────────────────────────────────────────
            "gemma" => load(config, |cfg: gemma::Config| {
                gemma::Model::new(use_flash_attn, &cfg, vb)
            }),

            "gemma2" => load(config, |cfg: gemma2::Config| {
                gemma2::Model::new(use_flash_attn, &cfg, vb)
            }),

            "gemma3" | "gemma3_text" => load(config, |cfg: gemma3::Config| {
                gemma3::Model::new(use_flash_attn, &cfg, vb)
            }),

            // ── Other decoder-only ─────────────────────────────────────────
            "starcoder2" => load(config, |cfg: starcoder2::Config| {
                starcoder2::Model::new(&cfg, vb)
            }),

            "deepseek_v2" | "deepseek-v2" => load(config, |cfg: deepseek2::DeepSeekV2Config| {
                deepseek2::DeepSeekV2::new(&cfg, vb)
            }),

            "olmo" => load(config, |cfg: olmo::Config| olmo::Model::new(&cfg, vb)),

            "olmo2" => load(config, |cfg: olmo2::Config| olmo2::Model::new(&cfg, vb)),

            "stablelm" | "stablelm_epoch" => load(config, |cfg: stable_lm::Config| {
                stable_lm::Model::new(&cfg, vb)
            }),

            "yi" => load(config, |cfg: yi::Config| yi::Model::new(&cfg, vb)),

            "helium" => load(config, |mut cfg: helium::Config| {
                cfg.use_flash_attn = use_flash_attn;
                helium::Model::new(&cfg, vb)
            }),

            "granite" => load(config, |cfg: granite::Config| {
                granite::GraniteForCausalLM::load(vb, &cfg)
            }),

            "chatglm" => load(config, |cfg: chatglm::Config| {
                chatglm::Model::new(&cfg, vb)
            }),

            _ => Err(candle::Error::Msg(format!(
                "unsupported model type '{}' for safetensors loading. \
                 Supported: {}",
                config.model_type,
                SUPPORTED_FLOAT_MODELS.join(", ")
            ))),
        }
    }

    // ------------------------------------------------------------------
    // Internal: GGUF model dispatch
    // ------------------------------------------------------------------

    fn load_gguf(path: &Path, model_type: &str, device: &Device) -> Result<Box<dyn CausalLM>> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| candle::Error::Msg(format!("cannot open GGUF file: {e}")))?;
        let content = candle::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| candle::Error::Msg(format!("failed to read GGUF: {e}")))?;
        Self::load_gguf_from_content(content, &mut file, model_type, device)
    }

    fn load_gguf_from_content<R: std::io::Seek + std::io::Read>(
        content: candle::quantized::gguf_file::Content,
        reader: &mut R,
        arch: &str,
        device: &Device,
    ) -> Result<Box<dyn CausalLM>> {
        match arch.to_lowercase().as_str() {
            "llama" => Ok(Box::new(quantized_llama::ModelWeights::from_gguf(
                content, reader, device,
            )?)),

            "phi" | "phi-msft" => Ok(Box::new(quantized_phi::ModelWeights::from_gguf(
                content, reader, device,
            )?)),

            "phi3" => Ok(Box::new(quantized_phi3::ModelWeights::from_gguf(
                false, content, reader, device,
            )?)),

            "qwen2" => Ok(Box::new(quantized_qwen2::ModelWeights::from_gguf(
                content, reader, device,
            )?)),

            "qwen3" => Ok(Box::new(quantized_qwen3::ModelWeights::from_gguf(
                content, reader, device,
            )?)),

            "gemma" | "gemma3" => Ok(Box::new(quantized_gemma3::ModelWeights::from_gguf(
                content, reader, device,
            )?)),

            _ => Err(candle::Error::Msg(format!(
                "unsupported GGUF architecture '{}'. Supported: {}",
                arch,
                SUPPORTED_GGUF_MODELS.join(", ")
            ))),
        }
    }
}
