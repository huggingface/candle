use hf_hub::{
    api::sync::{Api, ApiError},
    Repo,
};
use std::path::PathBuf;

pub fn download_safetensors(model_id: &str, revision_id: &str) -> Result<PathBuf, ApiError> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.into(),
        hf_hub::RepoType::Model,
        revision_id.into(),
    ));
    let weights = repo.get("model.safetensors")?;
    Ok(weights)
}

pub fn download_tokenizer_config(model_id: &str, revision_id: &str) -> Result<PathBuf, ApiError> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.into(),
        hf_hub::RepoType::Model,
        revision_id.into(),
    ));
    let weights = repo.get("tokenizer_config.json")?;
    Ok(weights)
}

pub fn download_special_map_config(model_id: &str, revision_id: &str) -> Result<PathBuf, ApiError> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.into(),
        hf_hub::RepoType::Model,
        revision_id.into(),
    ));
    let weights = repo.get("special_tokens_map.json")?;
    Ok(weights)
}

pub fn download_tokenizer(model_id: &str, revision_id: &str) -> Result<PathBuf, ApiError> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.into(),
        hf_hub::RepoType::Model,
        revision_id.into(),
    ));
    let weights = repo.get("tokenizer.json")?;
    Ok(weights)
}

pub fn download_config(model_id: &str, revision_id: &str) -> Result<PathBuf, ApiError> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.into(),
        hf_hub::RepoType::Model,
        revision_id.into(),
    ));

    let config = repo.get("config.json")?;
    Ok(config)
}
