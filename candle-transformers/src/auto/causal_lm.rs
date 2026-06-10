use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::path::{Path, PathBuf};

use super::config::{AutoConfig, Weights};
use crate::models::{
    chatglm, codegeex4_9b, deepseek2, falcon, gemma, gemma2, gemma3, glm4, helium, llama, mistral,
    mixtral, olmo, olmo2, phi3, qwen2, qwen2_moe, qwen3, qwen3_moe, stable_lm, starcoder2, yi,
};
use crate::models::{
    quantized_gemma3, quantized_glm4, quantized_lfm2, quantized_llama, quantized_phi,
    quantized_phi3, quantized_qwen2, quantized_qwen3, quantized_qwen3_moe,
};

/// Opaque, cloneable snapshot of a model's KV cache state.
pub trait CacheSnapshot: Send + Sync {
    fn clone_box(&self) -> Box<dyn CacheSnapshot>;
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Concrete snapshot: one `Option<(key, value)>` pair per transformer layer.
pub struct LayerKvSnapshot(pub Vec<Option<(Tensor, Tensor)>>);

impl CacheSnapshot for LayerKvSnapshot {
    fn clone_box(&self) -> Box<dyn CacheSnapshot> {
        Box::new(LayerKvSnapshot(self.0.clone()))
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A causal language model that can run autoregressive inference.
pub trait CausalLM: Send {
    fn model_type(&self) -> &'static str;

    /// Run one forward step. Returns logits for the last input token.
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>;

    /// Reset the internal KV cache.
    fn clear_kv_cache(&mut self) {}

    fn snapshot_cache(&self) -> Option<Box<dyn CacheSnapshot>> {
        None
    }

    fn restore_cache(&mut self, _snap: &dyn CacheSnapshot) {}
}

#[derive(Debug, Clone)]
pub enum QuantizationFormat {
    Gguf { path: PathBuf },
}

#[derive(Debug, Clone, Default)]
pub struct AutoModelOptions {
    pub revision: Option<String>,
    pub dtype: Option<DType>,
    pub use_flash_attn: bool,
    pub quantization: Option<QuantizationFormat>,
}

fn default_dtype(device: &Device) -> DType {
    if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Factory for loading causal language models.
pub struct AutoModelForCausalLM;

impl AutoModelForCausalLM {
    /// Load a model from the HuggingFace Hub.
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
        Self::load_float_model(&config, vb, options.use_flash_attn)
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
        Self::load_float_model(&config, vb, options.use_flash_attn)
    }

    /// Load directly from a local `.gguf` file.
    pub fn from_gguf(path: &Path, device: &Device) -> Result<Box<dyn CausalLM>> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| candle::Error::Msg(format!("cannot open GGUF file: {e}")))?;
        let content = candle::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| candle::Error::Msg(format!("failed to read GGUF: {e}")))?;
        let arch = content
            .metadata
            .get("general.architecture")
            .ok_or_else(|| candle::Error::Msg("GGUF missing general.architecture".to_string()))?
            .to_string()
            .map_err(|_| candle::Error::Msg("general.architecture is not a string".to_string()))?
            .clone();
        Self::load_gguf_from_content(content, &mut file, &arch, device)
    }

    fn load_float_model(
        config: &AutoConfig,
        vb: VarBuilder,
        use_flash_attn: bool,
    ) -> Result<Box<dyn CausalLM>> {
        crate::make_auto_map!(config, vb, {
                "llama"          => (llama::LlamaConfig,           |cfg: llama::LlamaConfig, vb: VarBuilder| llama::LlamaForCausalLM::new(&cfg, vb)),
                "yi"             => (yi::YiConfig,                 |cfg: yi::YiConfig, vb: VarBuilder| yi::Model::from_config(&cfg, vb)),
                "mistral"        => (mistral::Config,              |cfg: mistral::Config, vb: VarBuilder| mistral::Model::new(&cfg, vb)),
                "mixtral"        => (mixtral::Config,              |cfg: mixtral::Config, vb: VarBuilder| mixtral::Model::new(&cfg, vb)),
                "phi3"           => (phi3::Config,                 |cfg: phi3::Config, vb: VarBuilder| phi3::Model::new(&cfg, vb)),
                "qwen2"          => (qwen2::Config,               |cfg: qwen2::Config, vb: VarBuilder| qwen2::ModelForCausalLM::new(&cfg, vb)),
                "qwen2_moe"      => (qwen2_moe::Config,           |cfg: qwen2_moe::Config, vb: VarBuilder| qwen2_moe::Model::new(&cfg, vb)),
                "qwen3"          => (qwen3::Config,               |cfg: qwen3::Config, vb: VarBuilder| qwen3::ModelForCausalLM::new(&cfg, vb)),
                "qwen3_moe"      => (qwen3_moe::Config,           |cfg: qwen3_moe::Config, vb: VarBuilder| qwen3_moe::ModelForCausalLM::new(&cfg, vb)),
                "gemma"          => (gemma::Config,                |cfg: gemma::Config, vb: VarBuilder| gemma::Model::new(use_flash_attn, &cfg, vb)),
                "gemma2"         => (gemma2::Config,               |cfg: gemma2::Config, vb: VarBuilder| gemma2::Model::new(use_flash_attn, &cfg, vb)),
                "gemma3"         => (gemma3::Config,               |cfg: gemma3::Config, vb: VarBuilder| gemma3::Model::new(use_flash_attn, &cfg, vb)),
                "gemma3_text"    => (gemma3::Config,               |cfg: gemma3::Config, vb: VarBuilder| gemma3::Model::new(use_flash_attn, &cfg, vb)),
                "starcoder2"     => (starcoder2::Config,           |cfg: starcoder2::Config, vb: VarBuilder| starcoder2::Model::new(&cfg, vb)),
                "deepseek_v2"    => (deepseek2::DeepSeekV2Config,  |cfg: deepseek2::DeepSeekV2Config, vb: VarBuilder| deepseek2::DeepSeekV2::new(&cfg, vb)),
                "deepseek-v2"    => (deepseek2::DeepSeekV2Config,  |cfg: deepseek2::DeepSeekV2Config, vb: VarBuilder| deepseek2::DeepSeekV2::new(&cfg, vb)),
                "olmo"           => (olmo::Config,                 |cfg: olmo::Config, vb: VarBuilder| olmo::Model::new(&cfg, vb)),
                "olmo2"          => (olmo2::Config,                |cfg: olmo2::Config, vb: VarBuilder| olmo2::Model::new(&cfg, vb)),
                "stablelm"       => (stable_lm::Config,            |cfg: stable_lm::Config, vb: VarBuilder| stable_lm::Model::new(&cfg, vb)),
                "stablelm_epoch" => (stable_lm::Config,            |cfg: stable_lm::Config, vb: VarBuilder| stable_lm::Model::new(&cfg, vb)),
                "helium"         => (helium::Config,               |cfg: helium::Config, vb: VarBuilder| helium::Model::new(&cfg, vb)),
                "falcon"         => (falcon::Config,               |cfg: falcon::Config, vb: VarBuilder| falcon::Falcon::load(vb, cfg)),
                "chatglm"        => (chatglm::Config,              |cfg: chatglm::Config, vb: VarBuilder| chatglm::Model::new(&cfg, vb)),
                "glm4"           => (glm4::Config,                 |cfg: glm4::Config, vb: VarBuilder| glm4::Model::new(&cfg, vb)),
                "codegeex4"      => (codegeex4_9b::Config,         |cfg: codegeex4_9b::Config, vb: VarBuilder| codegeex4_9b::Model::new(&cfg, vb)),
        })
    }

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
            "llama"     => quantized_llama::ModelWeights,
            "phi"       => quantized_phi::ModelWeights,
            "phi3"      => quantized_phi3::ModelWeights,
            "qwen2"     => quantized_qwen2::ModelWeights,
            "qwen3"     => quantized_qwen3::ModelWeights,
            "qwen3_moe" => quantized_qwen3_moe::GGUFQWenMoE,
            "gemma"     => quantized_gemma3::ModelWeights,
            "gemma3"    => quantized_gemma3::ModelWeights,
            "lfm2"      => quantized_lfm2::ModelWeights,
            "glm4"      => quantized_glm4::ModelWeights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    struct MockModel {
        kv_cache: Vec<Option<(Tensor, Tensor)>>,
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

        fn forward(&mut self, input_ids: &Tensor, _seqlen_offset: usize) -> Result<Tensor> {
            self.call_count += 1;
            let dev = input_ids.device();
            for slot in self.kv_cache.iter_mut() {
                let k = Tensor::zeros((1, 1, 4), DType::F32, dev)?;
                let v = Tensor::full(self.call_count as f32, (1, 1, 4), dev)?;
                *slot = Some((k, v));
            }
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

    #[test]
    fn test_snapshot_and_restore() -> Result<()> {
        let dev = Device::Cpu;
        let mut model = MockModel::new(2);
        let ids = Tensor::zeros((1, 1), DType::U32, &dev)?;

        model.forward(&ids, 0)?;
        assert_eq!(read_v_scalar(&model, 0), 1.0);

        let snap = model.snapshot_cache().expect("mock always snapshots");
        model.forward(&ids, 1)?;
        assert_eq!(read_v_scalar(&model, 0), 2.0);

        model.restore_cache(snap.as_ref());
        assert_eq!(read_v_scalar(&model, 0), 1.0);
        Ok(())
    }

    #[test]
    fn test_default_no_snapshot() {
        struct NoopModel;
        impl CausalLM for NoopModel {
            fn model_type(&self) -> &'static str {
                "noop"
            }
            fn forward(&mut self, _ids: &Tensor, _offset: usize) -> Result<Tensor> {
                unimplemented!()
            }
        }
        assert!(NoopModel.snapshot_cache().is_none());
    }
}
