use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
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
        _dtype: DType,
        _device: &Device,
        use_flash_attn: bool,
    ) -> Result<Box<dyn CausalLM>> {
        crate::make_auto_map!(config, vb, use_flash_attn, {
            // Llama family
            "llama"          => (llama::LlamaConfig,              llama::LlamaForCausalLM),
            // Mistral / Mixtral
            "mistral"        => (mistral::Config,                 mistral::Model),
            "mixtral"        => (mixtral::Config,                 mixtral::Model),
            // Phi family
            "phi"            => (phi::Config,                     phi::Model),
            "phi-msft"       => (phi::Config,                     phi::Model),
            "phi3"           => (phi3::Config,                    phi3::Model),
            // Qwen family
            "qwen2"          => (qwen2::Config,                   qwen2::ModelForCausalLM),
            "qwen2_moe"      => (qwen2_moe::Config,               qwen2_moe::Model),
            "qwen3"          => (qwen3::Config,                   qwen3::ModelForCausalLM),
            "qwen3_moe"      => (qwen3_moe::Config,               qwen3_moe::ModelForCausalLM),
            // Gemma family
            "gemma"          => (gemma::Config,                   gemma::Model),
            "gemma2"         => (gemma2::Config,                  gemma2::Model),
            "gemma3"         => (gemma3::Config,                  gemma3::Model),
            "gemma3_text"    => (gemma3::Config,                  gemma3::Model),
            // Other decoder-only models
            "starcoder2"     => (starcoder2::Config,              starcoder2::Model),
            "deepseek_v2"    => (deepseek2::DeepSeekV2Config,     deepseek2::DeepSeekV2),
            "deepseek-v2"    => (deepseek2::DeepSeekV2Config,     deepseek2::DeepSeekV2),
            "olmo"           => (olmo::Config,                    olmo::Model),
            "olmo2"          => (olmo2::Config,                   olmo2::Model),
            "stablelm"       => (stable_lm::Config,               stable_lm::Model),
            "stablelm_epoch" => (stable_lm::Config,               stable_lm::Model),
            "yi"             => (yi::Config,                      yi::Model),
            "helium"         => (helium::Config,                  helium::Model),
            "granite"        => (granite::GraniteConfig,          granite::GraniteForCausalLM),
            "chatglm"        => (chatglm::Config,                 chatglm::Model),
        })
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
        crate::make_gguf_map!(arch, content, reader, device, {
            "llama"    => quantized_llama::ModelWeights,
            "phi"      => quantized_phi::ModelWeights,
            "phi-msft" => quantized_phi::ModelWeights,
            "phi3"     => quantized_phi3::ModelWeights,
            "qwen2"    => quantized_qwen2::ModelWeights,
            "qwen3"    => quantized_qwen3::ModelWeights,
            "gemma"    => quantized_gemma3::ModelWeights,
            "gemma3"   => quantized_gemma3::ModelWeights,
        })
    }
}
