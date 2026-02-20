use candle::{DType, Device, Result};
use candle_nn::VarBuilder;
use serde::Deserialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct RawConfig {
    model_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AutoConfig {
    pub model_type: String,
    pub raw_json: String,
}

impl AutoConfig {
    #[cfg(feature = "hf-hub")]
    pub fn from_pretrained(model_id: &str, revision: Option<&str>) -> Result<Self> {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        let revision = revision.unwrap_or("main");
        let api = Api::new().map_err(|e| candle::Error::Msg(format!("HF Hub error: {e}")))?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));

        let config_path = repo
            .get("config.json")
            .map_err(|e| candle::Error::Msg(format!("Failed to get config.json: {e}")))?;

        Self::from_local(&config_path)
    }

    pub fn from_local(config_path: &Path) -> Result<Self> {
        let raw_json = std::fs::read_to_string(config_path)?;
        let raw: RawConfig = serde_json::from_str(&raw_json)
            .map_err(|e| candle::Error::Msg(format!("Invalid config.json: {e}")))?;

        let model_type = raw.model_type.ok_or_else(|| {
            candle::Error::Msg("config.json missing 'model_type' field".to_string())
        })?;

        Ok(Self {
            model_type,
            raw_json,
        })
    }

    pub fn parse<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_str(&self.raw_json)
            .map_err(|e| candle::Error::Msg(format!("Failed to parse config: {e}")))
    }
}

pub struct Weights {
    paths: Vec<PathBuf>,
}

impl Weights {
    #[cfg(feature = "hf-hub")]
    pub fn from_pretrained(model_id: &str, revision: Option<&str>) -> Result<Self> {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        let revision = revision.unwrap_or("main");
        let api = Api::new().map_err(|e| candle::Error::Msg(format!("HF Hub error: {e}")))?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));

        let paths = Self::get_weight_files(&repo)?;
        Ok(Self { paths })
    }

    pub fn from_local(paths: Vec<PathBuf>) -> Self {
        Self { paths }
    }

    /// Create a VarBuilder from the loaded weights.
    ///
    /// # Safety
    /// Uses memory-mapped files. The files must not be modified while in use.
    pub unsafe fn into_var_builder(
        self,
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'static>> {
        VarBuilder::from_mmaped_safetensors(&self.paths, dtype, device)
    }

    #[cfg(feature = "hf-hub")]
    fn get_weight_files(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<PathBuf>> {
        // Try sharded model first (index file lists all weight shards)
        if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            return Self::load_sharded_weights(repo, &index_path);
        }

        // Fall back to single weight file
        if let Ok(path) = repo.get("model.safetensors") {
            return Ok(vec![path]);
        }

        Err(candle::Error::Msg(
            "No weights found. Expected 'model.safetensors' or 'model.safetensors.index.json'"
                .to_string(),
        ))
    }

    #[cfg(feature = "hf-hub")]
    fn load_sharded_weights(
        repo: &hf_hub::api::sync::ApiRepo,
        index_path: &Path,
    ) -> Result<Vec<PathBuf>> {
        let index_json = std::fs::read_to_string(index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_json).map_err(|e| {
            candle::Error::Msg(format!("Invalid model.safetensors.index.json: {e}"))
        })?;

        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| candle::Error::Msg("Index missing 'weight_map' field".to_string()))?;

        let files: HashSet<&str> = weight_map.values().filter_map(|v| v.as_str()).collect();

        if files.is_empty() {
            return Err(candle::Error::Msg(
                "Index 'weight_map' is empty".to_string(),
            ));
        }

        let mut paths = Vec::with_capacity(files.len());
        // TODO: Consider concurrent downloads for large sharded models
        for filename in files {
            let path = repo.get(filename).map_err(|e| {
                candle::Error::Msg(format!("Failed to get shard '{filename}': {e}"))
            })?;
            paths.push(path);
        }

        Ok(paths)
    }
}
