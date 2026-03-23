//! Model weight loading (safetensors + name mapping).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle::Device;
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::config::AsrConfig;

#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub dtype: candle::DType,
    pub use_flash_attn: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            dtype: candle::DType::F32,
            use_flash_attn: false,
        }
    }
}

pub fn load_model_from_pretrained(
    model_id_or_path: &str,
    device: &Device,
    opts: &LoadOptions,
) -> Result<(AsrConfig, super::AsrModel)> {
    validate_load_options(device, opts)?;

    let model_dir = resolve_model_dir(model_id_or_path)?;
    let config = load_config_json(&model_dir)?;
    let weights_paths = ensure_weights_files(model_id_or_path, &model_dir)?;

    let weight_refs: Vec<&Path> = weights_paths.iter().map(PathBuf::as_path).collect();
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(weight_refs.as_slice(), opts.dtype, device)? };
    let thinker = super::thinker::ThinkerForConditionalGeneration::load(
        &config.thinker_config,
        vb.pp("thinker"),
        device,
        opts.use_flash_attn,
    )?;

    Ok((
        config,
        super::AsrModel {
            model_dir,
            weights_paths,
            thinker,
        },
    ))
}

#[cfg(not(feature = "flash-attn"))]
fn validate_load_options(_device: &Device, opts: &LoadOptions) -> Result<()> {
    if opts.use_flash_attn {
        bail!(
            "flash-attn was requested but is not enabled in this build. Rebuild with `--features flash-attn`."
        );
    }
    Ok(())
}

#[cfg(feature = "flash-attn")]
fn validate_load_options(device: &Device, opts: &LoadOptions) -> Result<()> {
    if !opts.use_flash_attn {
        return Ok(());
    }

    if !device.is_cuda() {
        bail!(
            "flash-attn requires a CUDA device, got {:?}",
            device.location()
        );
    }

    match opts.dtype {
        candle::DType::F16 | candle::DType::BF16 => {}
        other => bail!("flash-attn requires dtype f16/bf16, got {other:?}"),
    }

    Ok(())
}

fn resolve_model_dir(model_id_or_path: &str) -> Result<PathBuf> {
    let path = Path::new(model_id_or_path);
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    use hf_hub::api::sync::Api;
    use hf_hub::{Repo, RepoType};

    let api = Api::new().context("failed to create HuggingFace API")?;
    let repo = api.repo(Repo::new(model_id_or_path.to_string(), RepoType::Model));
    let config_path = repo
        .get("config.json")
        .with_context(|| format!("failed to download config.json for {model_id_or_path:?}"))?;

    config_path.parent().map(Path::to_path_buf).ok_or_else(|| {
        anyhow::anyhow!(
            "failed to determine HuggingFace snapshot directory for {model_id_or_path:?}"
        )
    })
}

fn load_config_json(model_dir: &Path) -> Result<AsrConfig> {
    let config_path = model_dir.join("config.json");
    let data = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read config.json from {config_path:?}"))?;
    let cfg: AsrConfig = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse config.json from {config_path:?}"))?;
    Ok(cfg)
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: serde_json::Map<String, serde_json::Value>,
}

fn shard_filenames_from_index(index_path: &Path) -> Result<Vec<String>> {
    let data = std::fs::read_to_string(index_path)
        .with_context(|| format!("failed to read safetensors index from {index_path:?}"))?;
    let idx: SafetensorsIndex = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse safetensors index from {index_path:?}"))?;

    let mut shards: BTreeSet<String> = BTreeSet::new();
    for (param, filename) in idx.weight_map {
        let fname = filename.as_str().ok_or_else(|| {
            anyhow::anyhow!(
                "safetensors index weight_map value must be a string for key {param:?}: {index_path:?}"
            )
        })?;
        if fname.trim().is_empty() {
            bail!("safetensors index contains empty shard filename for key {param:?}");
        }
        shards.insert(fname.to_string());
    }

    if shards.is_empty() {
        bail!("safetensors index contains no shards: {index_path:?}");
    }

    Ok(shards.into_iter().collect())
}

fn shard_paths_from_local_index(model_dir: &Path, index_path: &Path) -> Result<Vec<PathBuf>> {
    let filenames = shard_filenames_from_index(index_path)?;
    let mut out: Vec<PathBuf> = Vec::with_capacity(filenames.len());
    for fname in filenames {
        let p = model_dir.join(fname);
        if !p.exists() {
            bail!("missing sharded safetensors file {p:?} referenced by {index_path:?}");
        }
        out.push(p);
    }
    Ok(out)
}

fn ensure_weights_files(model_id_or_path: &str, model_dir: &Path) -> Result<Vec<PathBuf>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        return shard_paths_from_local_index(model_dir, &index_path);
    }

    // Local dir but weights missing.
    if Path::new(model_id_or_path).exists() {
        bail!("no model.safetensors or model.safetensors.index.json found in {model_dir:?}");
    }

    // Remote: attempt to download either a single safetensors file or an index + shards.
    use hf_hub::api::sync::Api;
    use hf_hub::{Repo, RepoType};

    let api = Api::new().context("failed to create HuggingFace API")?;
    let repo = api.repo(Repo::new(model_id_or_path.to_string(), RepoType::Model));

    let single_res = repo.get("model.safetensors");
    if let Ok(p) = single_res {
        return Ok(vec![p]);
    }
    let single_err = single_res
        .err()
        .ok_or_else(|| anyhow::anyhow!("internal error: missing error for model.safetensors"))?;

    let index_path = repo.get("model.safetensors.index.json").with_context(|| {
        format!(
            "failed to download model.safetensors.index.json for {model_id_or_path:?} (also tried model.safetensors: {single_err:#})"
        )
    })?;

    let shard_filenames = shard_filenames_from_index(&index_path)?;
    let mut shard_paths: Vec<PathBuf> = Vec::with_capacity(shard_filenames.len());
    for fname in shard_filenames {
        let p = repo
            .get(fname.as_str())
            .with_context(|| format!("failed to download {fname:?} for {model_id_or_path:?}"))?;
        shard_paths.push(p);
    }
    Ok(shard_paths)
}

#[cfg(test)]
mod tests {
    use super::shard_filenames_from_index;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_tmp_dir(prefix: &str) -> anyhow::Result<PathBuf> {
        let mut dir = std::env::temp_dir();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| anyhow::anyhow!("system time before unix epoch: {e:?}"))?;
        dir.push(format!("{prefix}_{}", now.as_nanos()));
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    #[test]
    fn test_shard_filenames_from_index_dedupes_and_sorts() -> anyhow::Result<()> {
        let dir = unique_tmp_dir("qwen3_asr_shard_index")?;
        let index_path = dir.join("model.safetensors.index.json");

        let index = serde_json::json!({
            "weight_map": {
                "a": "model-00002-of-00002.safetensors",
                "b": "model-00001-of-00002.safetensors",
                "c": "model-00001-of-00002.safetensors"
            }
        });
        fs::write(&index_path, index.to_string())?;

        let got = shard_filenames_from_index(&index_path)?;
        let want = vec![
            "model-00001-of-00002.safetensors".to_string(),
            "model-00002-of-00002.safetensors".to_string(),
        ];
        if got != want {
            anyhow::bail!("unexpected shard list: got={got:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn test_ensure_weights_files_local_single_file() -> anyhow::Result<()> {
        let dir = unique_tmp_dir("qwen3_asr_weights_single")?;
        let weights_path = dir.join("model.safetensors");
        fs::write(&weights_path, [])?;

        let dir_s = dir.to_string_lossy();
        let got = super::ensure_weights_files(dir_s.as_ref(), &dir)?;
        let want = vec![weights_path];
        if got != want {
            anyhow::bail!("unexpected weights paths: got={got:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn test_ensure_weights_files_local_sharded_index() -> anyhow::Result<()> {
        let dir = unique_tmp_dir("qwen3_asr_weights_sharded")?;
        let index_path = dir.join("model.safetensors.index.json");

        let index = serde_json::json!({
            "weight_map": {
                "a": "model-00002-of-00002.safetensors",
                "b": "model-00001-of-00002.safetensors"
            }
        });
        fs::write(&index_path, index.to_string())?;

        let shard1 = dir.join("model-00001-of-00002.safetensors");
        let shard2 = dir.join("model-00002-of-00002.safetensors");
        fs::write(&shard1, [])?;
        fs::write(&shard2, [])?;

        let dir_s = dir.to_string_lossy();
        let got = super::ensure_weights_files(dir_s.as_ref(), &dir)?;
        let want = vec![shard1, shard2];
        if got != want {
            anyhow::bail!("unexpected weights paths: got={got:?} want={want:?}");
        }
        Ok(())
    }
}
