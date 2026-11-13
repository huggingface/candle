//! Local checkpoint loaders. No downloads or remote code execution. Native
//! model instances can also be passed directly to the adapter constructors.
use super::{
    mimi::Mimi,
    parler::{Parler, ParlerSamplingOptions},
    whisper::{Whisper, WhisperModel},
};
use crate::*;
use candle::{DType, Device};
use candle_nn::VarBuilder;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};
use tokenizers::Tokenizer;

pub fn model_type(directory: &Path) -> Result<String> {
    if !directory.join("config.json").exists() && directory.join("first_stage.meta.json").exists() {
        return Ok(if directory.join("first_stage_q4k.gguf").exists()
            && !directory.join("first_stage.safetensors").exists()
        {
            "quantized_metavoice"
        } else {
            "metavoice"
        }
        .into());
    }
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
    if let Some(family) = cfg["model_type"].as_str() {
        return Ok(family.into());
    }
    // Older native codec checkpoints predate Transformers' model_type key.
    if cfg.get("audio_num_codebooks").is_some() && cfg.get("backbone_flavor").is_some() {
        return Ok("csm".into());
    }
    if cfg.get("vq_strides").is_some() && cfg.get("encoder_rates").is_some() {
        return Ok("snac".into());
    }
    if cfg.get("target_bandwidths").is_some() && cfg.get("audio_channels").is_some() {
        return Ok("encodec".into());
    }
    if cfg.get("model_bitrate").is_some()
        && cfg.get("latent_dim").is_some()
        && cfg.get("num_codebooks").is_some()
    {
        return Ok("dac".into());
    }
    Err(Error::InvalidInput(
        "config.json is missing model_type; applications can set ModelSpec.family explicitly"
            .into(),
    ))
}
pub(crate) fn tokenizer(directory: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(directory.join("tokenizer.json"))
        .map_err(|e| Error::InvalidInput(e.to_string()))
}

/// Select a native submodel without executing checkpoint code. Dotted keys
/// traverse JSON objects and tensor prefixes only, never filesystem paths.
pub(crate) fn submodel_options(
    options: &serde_json::Value,
) -> Result<(Option<String>, Option<String>)> {
    if !options.is_null() && !options.is_object() {
        return Err(Error::InvalidInput(
            "model options must be a JSON object".into(),
        ));
    }
    let path = |name: &str| -> Result<Option<String>> {
        let Some(value) = options.get(name) else {
            return Ok(None);
        };
        let value = value
            .as_str()
            .ok_or_else(|| Error::InvalidInput(format!("{name} must be a dotted string")))?;
        if value.len() > 1024
            || value.split('.').any(|part| {
                part.is_empty()
                    || !part
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
            })
        {
            return Err(Error::InvalidInput(format!(
                "{name} must contain nonempty identifier components separated by dots"
            )));
        }
        Ok(Some(value.to_owned()))
    };
    Ok((path("config_key")?, path("weight_prefix")?))
}

pub(crate) fn submodel_config<'a>(
    root: &'a serde_json::Value,
    key: Option<&str>,
) -> Result<&'a serde_json::Value> {
    let mut selected = root;
    if let Some(key) = key {
        for component in key.split('.') {
            selected = selected.get(component).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "checkpoint configuration has no config_key {key:?}"
                ))
            })?;
        }
    }
    if !selected.is_object() {
        return Err(Error::InvalidInput(
            "selected checkpoint configuration must be an object".into(),
        ));
    }
    Ok(selected)
}
pub(crate) fn weights(directory: &Path) -> Result<Vec<PathBuf>> {
    let index = directory.join("model.safetensors.index.json");
    if !index.exists() {
        return Ok(vec![directory.join("model.safetensors")]);
    }
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(index)?)?;
    let map = index["weight_map"]
        .as_object()
        .ok_or_else(|| Error::InvalidInput("invalid safetensors weight_map".into()))?;
    let files: BTreeSet<_> = map
        .values()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| Error::InvalidInput("invalid safetensors shard filename".into()))
        })
        .collect::<Result<_>>()?;
    if files.is_empty() {
        return Err(Error::InvalidInput("empty weight_map".into()));
    }
    Ok(files.into_iter().map(|f| directory.join(f)).collect())
}
pub(crate) fn load_whisper(
    directory: &Path,
    device: &Device,
    options: &serde_json::Value,
) -> Result<Box<dyn AsrModel + Send>> {
    let kind = model_type(directory)?;
    if kind != "whisper" {
        return Err(Error::Unsupported(format!(
            "ASR checkpoint type {kind:?}; available: whisper"
        )));
    }
    let cfg = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
    let model = if directory.join("model.gguf").exists() {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            directory.join("model.gguf"),
            device,
        )?;
        WhisperModel::Quantized(
            candle_transformers::models::whisper::quantized_model::Whisper::load(&vb, cfg)?,
        )
    } else {
        // SAFETY: mmap is read-only. Callers must not modify checkpoint files
        // while the model holds these mappings.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weights(directory)?, DType::F32, device)?
        };
        WhisperModel::Normal(candle_transformers::models::whisper::model::Whisper::load(
            &vb, cfg,
        )?)
    };
    let filters = Whisper::default_mel_filters(model.config().num_mel_bins)?;
    let mut model = Whisper::new(model, tokenizer(directory)?, filters, device.clone())?;
    if let Some(max_tokens) = options.get("max_tokens").and_then(|v| v.as_u64()) {
        model.set_max_tokens(
            usize::try_from(max_tokens)
                .map_err(|_| Error::InvalidInput("max_tokens overflow".into()))?,
        )?;
    }
    Ok(Box::new(model))
}
pub(crate) fn load_parler(
    directory: &Path,
    device: &Device,
    options: &serde_json::Value,
) -> Result<Box<dyn TtsModel + Send>> {
    let sampling = ParlerSamplingOptions::from_model_options(options)?;
    let kind = model_type(directory)?;
    if kind != "parler_tts" {
        return Err(Error::Unsupported(format!(
            "TTS checkpoint type {kind:?}; available: parler_tts"
        )));
    }
    let cfg = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&weights(directory)?, DType::F32, device)? };
    let model = candle_transformers::models::parler_tts::Model::new(&cfg, vb)?;
    let mut model = Parler::new(model, &cfg, tokenizer(directory)?, device.clone())?;
    model.set_sampling_options(sampling)?;
    Ok(Box::new(model))
}
pub(crate) fn load_mimi(
    directory: &Path,
    device: &Device,
    input: AudioFormat,
    options: &serde_json::Value,
) -> Result<Box<dyn RealtimeModel + Send>> {
    let kind = model_type(directory)?;
    if kind != "mimi" {
        return Err(Error::Unsupported(format!(
            "codec checkpoint type {kind:?}; available: mimi"
        )));
    }
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&weights(directory)?, DType::F32, device)? };
    let cfg = mimi_config(directory, options)?;
    let model = candle_transformers::models::mimi::Model::new(cfg, vb)?;
    Ok(Box::new(Mimi::new(model, device.clone(), input)?))
}

pub fn asr(directory: &Path, device: &Device) -> Result<Box<dyn AsrModel + Send>> {
    super::registry::ModelRegistry::standard()
        .asr(&super::registry::ModelSpec::new(directory), device)
}
pub fn tts(directory: &Path, device: &Device) -> Result<Box<dyn TtsModel + Send>> {
    super::registry::ModelRegistry::standard()
        .tts(&super::registry::ModelSpec::new(directory), device)
}
pub fn codec(
    directory: &Path,
    device: &Device,
    input: AudioFormat,
) -> Result<Box<dyn RealtimeModel + Send>> {
    super::registry::ModelRegistry::standard().realtime(
        &super::registry::ModelSpec::new(directory).with_input(input),
        device,
    )
}
pub fn realtime(
    directory: &Path,
    device: &Device,
    input: AudioFormat,
) -> Result<Box<dyn RealtimeModel + Send>> {
    super::registry::ModelRegistry::standard().realtime(
        &super::registry::ModelSpec::new(directory).with_input(input),
        device,
    )
}

pub(crate) fn mimi_config(
    directory: &Path,
    options: &serde_json::Value,
) -> Result<candle_transformers::models::mimi::Config> {
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
    let count = options
        .get("num_codebooks")
        .or_else(|| json.get("num_quantizers"))
        .or_else(|| json.get("quantizer_n_q"))
        .or_else(|| json.get("num_codebooks"));
    let count = match count {
        None => 16,
        Some(v) => v
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|v| (1..=32).contains(v))
            .ok_or_else(|| Error::InvalidInput("Mimi num_codebooks must be in 1..32".into()))?,
    };
    let mut cfg = candle_transformers::models::mimi::Config::v0_1(Some(count));
    for (name, expected) in [
        ("sampling_rate", 24000),
        ("audio_channels", 1),
        ("hidden_size", 512),
        ("num_attention_heads", 8),
        ("num_hidden_layers", 8),
        ("codebook_size", 2048),
        ("codebook_dim", 256),
    ] {
        if let Some(actual) = json.get(name) {
            if actual.as_u64() != Some(expected) {
                return Err(Error::Unsupported(format!(
                    "Mimi v0.1 {name} must be {expected}, got {actual}"
                )));
            }
        }
    }
    if let Some(rate) = json.get("frame_rate") {
        if rate.as_f64() != Some(12.5) {
            return Err(Error::Unsupported(
                "Mimi v0.1 frame_rate must be 12.5".into(),
            ));
        }
    }
    if let Some(ratios) = json.get("upsampling_ratios") {
        if ratios != &serde_json::json!([8, 6, 5, 4]) {
            return Err(Error::Unsupported(
                "Mimi v0.1 requires upsampling_ratios [8,6,5,4]".into(),
            ));
        }
    }
    if let Some(max) = json.get("max_position_embeddings") {
        cfg.transformer.max_seq_len = max
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|v| *v > 0 && *v <= 8192)
            .ok_or_else(|| {
                Error::Unsupported("Mimi position capacity must be in 1..8192".into())
            })?;
    }
    Ok(cfg)
}
