use anyhow::{Context, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::{
    fs::{self, File},
    io::copy,
    path::Path,
};

pub fn download_model(model_and_revision: &str) -> Result<()> {
    let (model_id, revision) = match model_and_revision.split_once(":") {
        Some((model_id, revision)) => (model_id, revision),
        None => (model_and_revision, "main"),
    };
    let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?.to_string_lossy().to_string();
        let tokenizer = api.get("tokenizer.json")?.to_string_lossy().to_string();
        let weights = api.get("model.safetensors")?.to_string_lossy().to_string();
        (config, tokenizer, weights)
    };
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_CONFIG={config_filename}");
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_TOKENIZER={tokenizer_filename}");
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_WEIGHTS={weights_filename}");

    Ok(())
}
