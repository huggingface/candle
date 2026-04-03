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
// CacheSnapshot
// ---------------------------------------------------------------------------

/// Opaque, cloneable snapshot of a model's KV cache state.
///
/// Snapshots are produced by [`CausalLM::snapshot_cache`] and consumed by
/// [`CausalLM::restore_cache`].  They enable prefix caching: prefill a shared
/// system prompt once, snapshot the cache, then restore it before each new
/// user turn to avoid recomputing the prefix.
///
/// Tensor data is reference-counted inside candle, so `clone_box` is a shallow
/// clone — prefix K/V tensors are shared, not copied.
pub trait CacheSnapshot: Send {
    /// Produce an independent clone of this snapshot.
    fn clone_box(&self) -> Box<dyn CacheSnapshot>;
    /// Expose the concrete type for downcasting in [`CausalLM::restore_cache`].
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Concrete snapshot for the dominant KV-cache layout: one
/// `Option<(key, value)>` pair per transformer layer.
///
/// This covers both the internal-cache pattern used by most models (Mistral,
/// Qwen2, Phi, Gemma, …) and the external-cache wrapper pattern (Llama,
/// Granite).  Tensors are ref-counted so cloning is cheap.
pub struct LayerKvSnapshot(pub Vec<Option<(Tensor, Tensor)>>);

impl CacheSnapshot for LayerKvSnapshot {
    fn clone_box(&self) -> Box<dyn CacheSnapshot> {
        Box::new(LayerKvSnapshot(self.0.clone()))
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

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

    /// Snapshot the KV cache at the current sequence position.
    ///
    /// Returns `None` for models that have not opted into snapshot support.
    /// When `Some`, the snapshot can be passed to [`restore_cache`] to fork
    /// generation from the same prefix without re-running the prefill.
    fn snapshot_cache(&self) -> Option<Box<dyn CacheSnapshot>> {
        None
    }

    /// Restore a previously captured KV cache snapshot.
    ///
    /// After calling this, the next `forward` call will continue from exactly
    /// the sequence position at which the snapshot was taken.  Silently
    /// ignored if the snapshot type does not match the model's concrete cache
    /// type (i.e. the snapshot came from a different model).
    fn restore_cache(&mut self, _snap: &dyn CacheSnapshot) {}
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
            "llama"          => (llama::Config,              llama::LlamaForCausalLM),
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
            "granite"        => (granite::Config,          granite::GraniteForCausalLM),
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    // ------------------------------------------------------------------
    // Minimal mock model that tracks its KV cache as a flat Vec so we can
    // inspect state without touching real model weights.
    // ------------------------------------------------------------------

    struct MockModel {
        /// Simulated per-layer KV cache: each entry is Some((k, v)) after the
        /// first forward pass, None before.
        kv_cache: Vec<Option<(Tensor, Tensor)>>,
        /// Counts forward calls; baked into the value tensor so we can tell
        /// which generation "step" produced a given cache entry.
        call_count: u32,
    }

    impl MockModel {
        fn new(num_layers: usize) -> Self {
            Self {
                kv_cache: vec![None; num_layers],
                call_count: 0,
            }
        }
    }

    impl CausalLM for MockModel {
        fn model_type(&self) -> &'static str {
            "mock"
        }

        /// Stores `call_count` as the sole scalar value in each layer's value
        /// tensor so tests can confirm which forward pass populated the cache.
        fn forward(&mut self, input_ids: &Tensor, _seqlen_offset: usize) -> Result<Tensor> {
            self.call_count += 1;
            let dev = input_ids.device();
            for slot in self.kv_cache.iter_mut() {
                let k = Tensor::zeros((1, 1, 4), DType::F32, dev)?;
                let v = Tensor::full(self.call_count as f32, (1, 1, 4), dev)?;
                *slot = Some((k, v));
            }
            // Return fake logits (vocab_size = 8)
            Tensor::zeros((1, 8), DType::F32, dev)
        }

        fn clear_kv_cache(&mut self) {
            for slot in self.kv_cache.iter_mut() {
                *slot = None;
            }
        }

        fn snapshot_cache(&self) -> Option<Box<dyn CacheSnapshot>> {
            Some(Box::new(LayerKvSnapshot(self.kv_cache.clone())))
        }

        fn restore_cache(&mut self, snap: &dyn CacheSnapshot) {
            if let Some(s) = snap.as_any().downcast_ref::<LayerKvSnapshot>() {
                self.kv_cache = s.0.clone();
            }
        }
    }

    /// Extract the scalar stored in the value tensor of layer `layer_idx`.
    /// The mock stores `call_count` there so we can identify which step it is.
    fn read_v_scalar(model: &MockModel, layer_idx: usize) -> f32 {
        model.kv_cache[layer_idx]
            .as_ref()
            .expect("cache slot should be populated")
            .1
            .flatten_all()
            .unwrap()
            .get(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    // ------------------------------------------------------------------
    // Test 1: snapshot captures current state; restore brings it back
    // ------------------------------------------------------------------
    #[test]
    fn test_snapshot_and_restore() -> Result<()> {
        let dev = Device::Cpu;
        let mut model = MockModel::new(2);
        let ids = Tensor::zeros((1, 1), DType::U32, &dev)?;

        // Step 0 → call_count becomes 1 after first forward
        model.forward(&ids, 0)?;
        assert_eq!(read_v_scalar(&model, 0), 1.0);

        // Snapshot after step 0
        let snap = model.snapshot_cache().expect("mock always snapshots");

        // Step 1 → call_count becomes 2
        model.forward(&ids, 1)?;
        assert_eq!(read_v_scalar(&model, 0), 2.0);

        // Restore to post-step-0 state
        model.restore_cache(snap.as_ref());
        assert_eq!(
            read_v_scalar(&model, 0),
            1.0,
            "cache should be back to step-0 value after restore"
        );

        Ok(())
    }

    // ------------------------------------------------------------------
    // Test 2: clone_box produces an independent snapshot
    // ------------------------------------------------------------------
    #[test]
    fn test_clone_box_is_independent() -> Result<()> {
        let dev = Device::Cpu;
        let mut model = MockModel::new(2);
        let ids = Tensor::zeros((1, 1), DType::U32, &dev)?;

        model.forward(&ids, 0)?; // call_count = 1

        let snap = model.snapshot_cache().unwrap();
        let snap_clone = snap.clone_box();

        // Advance model further
        model.forward(&ids, 1)?; // call_count = 2

        // Restoring from the clone should also reset to step-0 value
        model.restore_cache(snap_clone.as_ref());
        assert_eq!(
            read_v_scalar(&model, 0),
            1.0,
            "clone_box snapshot should be independent of the original"
        );

        Ok(())
    }

    // ------------------------------------------------------------------
    // Test 3: prefix forking — two continuations from the same snapshot
    // produce independent caches
    // ------------------------------------------------------------------
    #[test]
    fn test_prefix_fork_independence() -> Result<()> {
        let dev = Device::Cpu;
        let mut model = MockModel::new(2);
        let ids = Tensor::zeros((1, 1), DType::U32, &dev)?;

        // Prefill (shared prefix)
        model.forward(&ids, 0)?; // call_count = 1
        let prefix_snap = model.snapshot_cache().unwrap();
        let prefix_snap2 = prefix_snap.clone_box();

        // Fork 1: generate one token
        model.restore_cache(prefix_snap.as_ref());
        model.forward(&ids, 1)?; // call_count = 2
        let fork1_v = read_v_scalar(&model, 0);

        // Fork 2: generate from the same prefix
        model.restore_cache(prefix_snap2.as_ref());
        model.forward(&ids, 1)?; // call_count = 3
        let fork2_v = read_v_scalar(&model, 0);

        // The two forks should differ (different call_counts → different value tensors)
        assert_ne!(
            fork1_v, fork2_v,
            "each fork should produce its own distinct cache"
        );

        Ok(())
    }

    // ------------------------------------------------------------------
    // Test 4: models that don't opt in return None
    // ------------------------------------------------------------------
    #[test]
    fn test_default_no_snapshot() {
        struct NoopModel;
        impl CausalLM for NoopModel {
            fn model_type(&self) -> &'static str {
                "noop"
            }
            fn forward(
                &mut self,
                _ids: &Tensor,
                _offset: usize,
            ) -> Result<Tensor> {
                unimplemented!()
            }
        }

        let m = NoopModel;
        assert!(
            m.snapshot_cache().is_none(),
            "models without snapshot support must return None"
        );
    }

    // ------------------------------------------------------------------
    // Test 5: LayerKvSnapshot as_any round-trips correctly
    // ------------------------------------------------------------------
    #[test]
    fn test_layer_kv_snapshot_as_any() -> Result<()> {
        let dev = Device::Cpu;
        let t = Tensor::zeros((1, 1, 4), DType::F32, &dev)?;
        let snap: Box<dyn CacheSnapshot> =
            Box::new(LayerKvSnapshot(vec![Some((t.clone(), t.clone())), None]));

        assert!(
            snap.as_any().downcast_ref::<LayerKvSnapshot>().is_some(),
            "LayerKvSnapshot must downcast cleanly from &dyn CacheSnapshot"
        );

        let cloned = snap.clone_box();
        assert!(
            cloned.as_any().downcast_ref::<LayerKvSnapshot>().is_some(),
            "clone_box result must also downcast"
        );

        Ok(())
    }
}
