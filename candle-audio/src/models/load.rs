//! Local checkpoint loaders. No downloads or remote code execution. Native
//! model instances can also be passed directly to the adapter constructors.
use super::{
    mimi::Mimi,
    parler::Parler,
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
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
    cfg["model_type"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| Error::InvalidInput("config.json is missing model_type".into()))
}
fn tokenizer(directory: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(directory.join("tokenizer.json"))
        .map_err(|e| Error::InvalidInput(e.to_string()))
}
fn weights(directory: &Path) -> Result<Vec<PathBuf>> {
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
pub fn asr(directory: &Path, device: &Device) -> Result<Box<dyn AsrModel + Send>> {
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
    Ok(Box::new(Whisper::new(
        model,
        tokenizer(directory)?,
        filters,
        device.clone(),
    )?))
}
pub fn tts(directory: &Path, device: &Device) -> Result<Box<dyn TtsModel + Send>> {
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
    Ok(Box::new(Parler::new(
        model,
        &cfg,
        tokenizer(directory)?,
        device.clone(),
    )?))
}
pub fn codec(
    directory: &Path,
    device: &Device,
    input: AudioFormat,
) -> Result<Box<dyn RealtimeModel + Send>> {
    let kind = model_type(directory)?;
    if kind != "mimi" {
        return Err(Error::Unsupported(format!(
            "codec checkpoint type {kind:?}; available: mimi"
        )));
    }
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&weights(directory)?, DType::F32, device)? };
    let cfg = candle_transformers::models::mimi::Config::v0_1(None);
    let model = candle_transformers::models::mimi::Model::new(cfg, vb)?;
    Ok(Box::new(Mimi::new(model, device.clone(), input)?))
}
