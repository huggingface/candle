//! Model and configuration loading utilities.

use std::path::{Path, PathBuf};

use candle::{Device, Result};
use candle_nn::VarBuilder;
use candle_transformers::models::{LoadableModel, ModelConfig};
use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};
use tokenizers::Tokenizer;

pub mod defaults {
    pub const CONFIG_FILENAME: &str = "config.json";
    pub const TOKENIZER_FILENAME: &str = "tokenizer.json";
    pub const MODEL_PT_FILENAME: &str = "pytorch_model.bin";
    pub const MODEL_ST_FILENAME: &str = "model.safetensors";
    pub const ST_FILE_EXTENSION: &str = "safetensors";
}

/// Loads tokenizer, config, and model in one call.
///
/// This is a convenience function that combines loading of all three components
/// needed to run a model.
///
/// # Example  
/// ```ignore
/// use candle_utils::loader::load_tokenizer_config_model;
/// use candle_transformers::models::bert::{BertModel, Config};
///
/// let (tokenizer, config, model) = load_tokenizer_config_model::<BertModel, Config>(
///     &device,
///     "bert-base-uncased",
///     "main",
///     candle::DType::F32,
///     None,
///     None,
///     None,
/// )?;
/// ```
pub fn load_tokenizer_config_model<M, C>(
    device: &Device,
    model_id: &str,
    revision: &str,
    dtype: candle::DType,
    tokenizer_file: Option<&Path>,
    config_file: Option<&Path>,
    weights_file: Option<&Path>,
) -> Result<(Tokenizer, C, M)>
where
    M: LoadableModel<C>,
    C: ModelConfig,
{
    let repo = load_repo(model_id, revision)?;
    let tokenizer = load_tokenizer(&repo, tokenizer_file)?;
    let config = load_config(&repo, config_file)?;
    let model = load_model(&repo, device, dtype, &config, weights_file)?;

    Ok((tokenizer, config, model))
}

/// Creates a HuggingFace API repository handle.
pub fn load_repo(model_id: &str, revision: &str) -> Result<ApiRepo> {
    let api =
        Api::new().map_err(|e| candle::Error::Msg(format!("Couldn't create API. Error: {}", e)))?;

    Ok(api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    )))
}

/// Loads a tokenizer from the repository or local file.
pub fn load_tokenizer(repo: &ApiRepo, tokenizer_file: Option<&Path>) -> Result<Tokenizer> {
    let tokenizer_filepath = match tokenizer_file {
        Some(file) => PathBuf::from(file),
        None => repo.get(defaults::TOKENIZER_FILENAME).map_err(|e| {
            candle::Error::Msg(format!(
                "Couldn't load {}. Error: {}",
                defaults::TOKENIZER_FILENAME,
                e
            ))
        })?,
    };

    Tokenizer::from_file(tokenizer_filepath)
        .map_err(|e| candle::Error::Msg(format!("Couldn't load tokenizer. Error: {}", e)))
}

/// Loads model configuration from the repository or local file.
pub fn load_config<C>(repo: &ApiRepo, config_file: Option<&Path>) -> Result<C>
where
    C: ModelConfig,
{
    let config_filepath = match config_file {
        Some(file) => PathBuf::from(file),
        None => repo.get(defaults::CONFIG_FILENAME).map_err(|e| {
            candle::Error::Msg(format!(
                "Couldn't load {}. Error: {}",
                defaults::CONFIG_FILENAME,
                e
            ))
        })?,
    };

    let config_str = std::fs::read_to_string(config_filepath)?;

    let config: C = serde_json::from_str(&config_str)
        .map_err(|e| candle::Error::Msg(format!("Couldn't parse config. Error: {}", e)))?;

    Ok(config)
}

/// Loads model weights from the repository or local file.
///
/// Automatically detects and handles both SafeTensors and PyTorch weight formats,
/// preferring SafeTensors when available.
pub fn load_model<M, C>(
    repo: &ApiRepo,
    device: &Device,
    dtype: candle::DType,
    config: &C,
    weights_file: Option<&Path>,
) -> Result<M>
where
    M: LoadableModel<C>,
    C: ModelConfig,
{
    let weights_filename = match weights_file {
        Some(file) => PathBuf::from(file),
        None => {
            // Try safetensors first, then pytorch
            repo.get(defaults::MODEL_ST_FILENAME)
                .or_else(|_| repo.get(defaults::MODEL_PT_FILENAME))
                .map_err(|e| {
                    candle::Error::Msg(format!(
                        "Model weights not found. Expected '{}' or '{}'. Error: {}",
                        defaults::MODEL_ST_FILENAME,
                        defaults::MODEL_PT_FILENAME,
                        e
                    ))
                })?
        }
    };

    let vb = if weights_filename.extension().and_then(|s| s.to_str())
        == Some(defaults::ST_FILE_EXTENSION)
    {
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, device)? }
    } else {
        println!("Loading weights from {}", defaults::MODEL_PT_FILENAME);
        VarBuilder::from_pth(&weights_filename, dtype, device)?
    };

    M::load(vb, config)
}
