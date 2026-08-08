//! Directory-level loading for NVIDIA ModelOpt NVFP4 checkpoint exports.
//!
//! A ModelOpt NVFP4 export mixes dense, block-wise FP8, and packed NVFP4
//! tensors, and can lay them out as either a single `model.safetensors` or
//! several shards indexed by `model.safetensors.index.json`. Telling such a
//! checkpoint apart from a plain dense one, resolving its shard paths, and
//! parsing `config.json` into a model's own `Config` type were, before this
//! module, reimplemented by every consumer (huggingface/candle#3866): a
//! checkpoint's `config.json` carries fields a model's `Config` doesn't
//! need, but most `Config` types already derive `serde::Deserialize`
//! directly against the same HF-format `config.json` ModelOpt emits, so no
//! per-model translation layer is actually required -- consumers just
//! needed the detection and shard-resolution work done once.
//! [`ModelOptCheckpoint`] wraps those steps behind one directory-level API:
//!
//! ```no_run
//! # #[derive(serde::Deserialize)]
//! # struct Config { hidden_size: usize }
//! # fn run() -> candle::Result<()> {
//! use candle_transformers::modelopt_checkpoint::ModelOptCheckpoint;
//!
//! let ckpt = ModelOptCheckpoint::open("path/to/checkpoint")?;
//! let cfg: Config = ckpt.model_config()?;
//! // Safety: the shard files must outlive the returned `VarBuilder`
//! // unmodified -- see `VarBuilder::from_mmaped_safetensors`.
//! let vb = unsafe { ckpt.var_builder(candle::DType::BF16, &candle::Device::Cpu)? };
//! # let _ = (cfg, vb);
//! # Ok(())
//! # }
//! ```

use candle::{DType, Device, Result};
use candle_nn::VarBuilder;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// A directory holding an NVIDIA ModelOpt export: `config.json` plus one or
/// more `.safetensors` shards, optionally indexed by
/// `model.safetensors.index.json`.
#[derive(Debug, Clone)]
pub struct ModelOptCheckpoint {
    dir: PathBuf,
    config: serde_json::Value,
    shard_paths: Vec<PathBuf>,
}

impl ModelOptCheckpoint {
    /// Opens `dir`, reading `config.json` and resolving its safetensors
    /// shard(s) -- from `model.safetensors.index.json` if present, otherwise
    /// a single `model.safetensors` -- without loading any tensor data.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let config = read_json(&dir.join("config.json"))?;
        let shard_paths = resolve_shard_paths(&dir)?;
        Ok(Self {
            dir,
            config,
            shard_paths,
        })
    }

    /// The checkpoint directory passed to [`open`](Self::open).
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The resolved safetensors shard paths, deduplicated and sorted for a
    /// deterministic load order.
    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.shard_paths
    }

    /// True if this checkpoint is a ModelOpt NVFP4 export: either
    /// `config.json` names an FP4 algorithm under `quantization_config`
    /// (`quant_algo` or `quant_method` -- ModelOpt always sets one of
    /// these), or, for exports that only carry quantization metadata
    /// per-tensor, any shard has a `weight_scale_2` tensor -- an NVFP4 layer
    /// always carries one alongside its packed weight, and neither a dense
    /// nor a block-wise-FP8 layer does. Reading shard tensor names only
    /// deserializes each file's safetensors header, not its weight data.
    pub fn is_nvfp4(&self) -> Result<bool> {
        if config_declares_nvfp4(&self.config) {
            return Ok(true);
        }
        for path in &self.shard_paths {
            // Safety: only the header is deserialized here; see
            // MmapedSafetensors::new's own safety note.
            let shard = unsafe { candle::safetensors::MmapedSafetensors::new(path)? };
            if shard
                .tensors()
                .iter()
                .any(|(name, _)| name.ends_with("weight_scale_2"))
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Deserializes `config.json` into `T`, e.g. a model's own `Config`
    /// type. ModelOpt's `config.json` is HF-format plus a
    /// `quantization_config` block; any `T` that doesn't declare that field
    /// (most model configs don't) simply ignores it, same as
    /// `serde_json::from_reader` would on a plain HF `config.json`.
    pub fn model_config<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_value(self.config.clone())
            .map_err(|e| candle::Error::Msg(format!("modelopt: deserializing config.json: {e}")))
    }

    /// Builds a [`VarBuilder`] over every resolved shard, memory-mapped.
    ///
    /// # Safety
    ///
    /// Inherits [`VarBuilder::from_mmaped_safetensors`]'s safety
    /// requirement: the shard files must not be modified for as long as the
    /// returned `VarBuilder`, or any tensor loaded from it, stays alive.
    pub unsafe fn var_builder(&self, dtype: DType, device: &Device) -> Result<VarBuilder<'_>> {
        VarBuilder::from_mmaped_safetensors(&self.shard_paths, dtype, device)
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let bytes = std::fs::read(path)
        .map_err(|e| candle::Error::Msg(format!("modelopt: reading {path:?}: {e}")))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| candle::Error::Msg(format!("modelopt: parsing {path:?}: {e}")))
}

/// Resolves shard paths from `model.safetensors.index.json`'s `weight_map`
/// if present, deduplicated (the same shard backs many tensor names) and
/// sorted for a deterministic order; otherwise falls back to a single
/// `model.safetensors`.
fn resolve_shard_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        let index = read_json(&index_path)?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                candle::Error::Msg(format!("modelopt: {index_path:?} has no weight_map"))
            })?;
        let mut files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        files.sort();
        Ok(files.into_iter().map(|f| dir.join(f)).collect())
    } else {
        let single = dir.join("model.safetensors");
        if !single.is_file() {
            candle::bail!("modelopt: neither {index_path:?} nor {single:?} exists in {dir:?}")
        }
        Ok(vec![single])
    }
}

fn config_declares_nvfp4(config: &serde_json::Value) -> bool {
    config
        .get("quantization_config")
        .and_then(|q| q.get("quant_algo").or_else(|| q.get("quant_method")))
        .and_then(|v| v.as_str())
        .is_some_and(|s| s.to_ascii_uppercase().contains("FP4"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, Tensor};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A directory under the OS temp dir that removes itself on drop, so a
    /// failing assertion doesn't leak fixture files across test runs.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "candle-modelopt-checkpoint-test-{name}-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_json(path: &Path, value: &serde_json::Value) {
        std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
    }

    fn write_dense_safetensors(path: &Path, tensor_name: &str) {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        tensor.save_safetensors(tensor_name, path).unwrap();
    }

    #[test]
    fn open_resolves_a_single_unshareded_checkpoint() {
        let dir = TempDir::new("single");
        write_json(
            &dir.path().join("config.json"),
            &serde_json::json!({"hidden_size": 16}),
        );
        write_dense_safetensors(&dir.path().join("model.safetensors"), "layer.weight");

        let ckpt = ModelOptCheckpoint::open(dir.path()).unwrap();
        assert_eq!(ckpt.shard_paths(), &[dir.path().join("model.safetensors")]);
        assert!(!ckpt.is_nvfp4().unwrap());
    }

    #[test]
    fn open_missing_checkpoint_files_errors() {
        let dir = TempDir::new("missing");
        let err = ModelOptCheckpoint::open(dir.path()).unwrap_err();
        assert!(err.to_string().contains("config.json"));
    }

    #[test]
    fn open_resolves_sharded_checkpoint_from_index() {
        let dir = TempDir::new("sharded");
        write_json(
            &dir.path().join("config.json"),
            &serde_json::json!({"hidden_size": 16}),
        );
        write_dense_safetensors(
            &dir.path().join("model-00001.safetensors"),
            "layer.a.weight",
        );
        write_dense_safetensors(
            &dir.path().join("model-00002.safetensors"),
            "layer.b.weight",
        );
        write_json(
            &dir.path().join("model.safetensors.index.json"),
            &serde_json::json!({
                "metadata": {"total_size": 0},
                "weight_map": {
                    "layer.a.weight": "model-00001.safetensors",
                    "layer.b.weight": "model-00002.safetensors",
                },
            }),
        );

        let ckpt = ModelOptCheckpoint::open(dir.path()).unwrap();
        assert_eq!(
            ckpt.shard_paths(),
            &[
                dir.path().join("model-00001.safetensors"),
                dir.path().join("model-00002.safetensors"),
            ]
        );
    }

    #[test]
    fn is_nvfp4_detects_from_config_quantization_block() {
        let dir = TempDir::new("config-flagged");
        write_json(
            &dir.path().join("config.json"),
            &serde_json::json!({
                "hidden_size": 16,
                "quantization_config": {"quant_algo": "NVFP4"},
            }),
        );
        write_dense_safetensors(&dir.path().join("model.safetensors"), "layer.weight");

        let ckpt = ModelOptCheckpoint::open(dir.path()).unwrap();
        assert!(ckpt.is_nvfp4().unwrap());
    }

    #[test]
    fn is_nvfp4_detects_from_shard_tensor_names_without_config_hint() {
        let dir = TempDir::new("tensor-flagged");
        write_json(
            &dir.path().join("config.json"),
            &serde_json::json!({"hidden_size": 16}),
        );
        let device = Device::Cpu;
        let tensors: std::collections::HashMap<String, Tensor> = [
            (
                "layer.weight".to_string(),
                Tensor::zeros((2, 1), DType::U8, &device).unwrap(),
            ),
            (
                "layer.weight_scale_2".to_string(),
                Tensor::from_vec(vec![0.5f32], (), &device).unwrap(),
            ),
        ]
        .into_iter()
        .collect();
        candle::safetensors::save(&tensors, dir.path().join("model.safetensors")).unwrap();

        let ckpt = ModelOptCheckpoint::open(dir.path()).unwrap();
        assert!(ckpt.is_nvfp4().unwrap());
    }

    #[test]
    fn model_config_ignores_fields_the_target_type_does_not_declare() {
        #[derive(serde::Deserialize)]
        struct Small {
            hidden_size: usize,
        }

        let dir = TempDir::new("model-config");
        write_json(
            &dir.path().join("config.json"),
            &serde_json::json!({
                "hidden_size": 32,
                "quantization_config": {"quant_algo": "NVFP4", "block_size": 16},
            }),
        );
        write_dense_safetensors(&dir.path().join("model.safetensors"), "layer.weight");

        let ckpt = ModelOptCheckpoint::open(dir.path()).unwrap();
        let cfg: Small = ckpt.model_config().unwrap();
        assert_eq!(cfg.hidden_size, 32);
    }

    #[test]
    fn var_builder_loads_tensors_from_resolved_shards() {
        let dir = TempDir::new("var-builder");
        write_json(
            &dir.path().join("config.json"),
            &serde_json::json!({"hidden_size": 16}),
        );
        write_dense_safetensors(&dir.path().join("model.safetensors"), "layer.weight");

        let ckpt = ModelOptCheckpoint::open(dir.path()).unwrap();
        let vb = unsafe { ckpt.var_builder(DType::F32, &Device::Cpu).unwrap() };
        let weight = vb.get((2, 2), "layer.weight").unwrap();
        assert_eq!(weight.dims(), &[2, 2]);
    }
}
