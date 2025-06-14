//! Helper utils for common Candle use cases.
//!
//! This crate provides convenient utilities for working with Candle.

pub use device::get as get_device;
pub use tensor::normalize_l2;

/// Device management utilities.
pub mod device {
    use candle::{Device, Result};

    /// Helper function to get a single candle Device. This will not perform any multi device mapping.
    /// Will prioritize Metal or CUDA (if Candle is compiled with those features) and fallback to CPU.
    ///
    /// # Arguments
    /// * `use_cpu` - Force CPU usage even if GPU is available
    /// * `quiet` - Suppress informational messages about GPU availability
    ///
    /// # Example  
    /// ```
    /// let device = candle_utils::device::get(true, false).unwrap();
    /// ```
    pub fn get(use_cpu: bool, quiet: bool) -> Result<Device> {
        if use_cpu {
            Ok(Device::Cpu)
        } else if candle::utils::cuda_is_available() {
            Ok(Device::new_cuda(0)?)
        } else if candle::utils::metal_is_available() {
            Ok(Device::new_metal(0)?)
        } else {
            if !quiet {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                {
                    println!(
                        "Running on CPU, to run on GPU (metal), build with `--features metal`"
                    );
                }

                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                {
                    println!("Running on CPU, to run on GPU, build with `--features cuda`");
                }
            }

            Ok(Device::Cpu)
        }
    }
}

/// Tensor operation utilities.
pub mod tensor {
    use candle::{Result, Tensor};

    /// Perform L2 normalization along the last dimension.
    ///
    /// This function normalizes each row vector to have unit L2 norm, which is commonly
    /// used in embedding and similarity computations.
    ///
    /// # Example  
    /// ```
    /// use candle_core::Tensor;
    /// use candle_utils::tensor::normalize_l2;
    ///
    /// let device = candle_utils::get_device(true, false).unwrap();
    /// let x = Tensor::new(&[[0f32, 1.], [2., 3.]], &device).unwrap();
    /// let normalized_x = normalize_l2(&x).unwrap();
    /// ```
    pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
        let squared = &v.sqr()?;
        let summed = &squared.sum_keepdim(1)?;
        let norms = &summed.sqrt()?;
        v.broadcast_div(norms)
    }
}

/// Tokenization utilities for working with tokenizers.
pub mod tokenization {
    use candle::{Device, Result, Tensor};
    use tokenizers::{Encoding, Tokenizer};

    /// Encodes input strings into tokens, token IDs, and type IDs for model input.
    ///
    /// This function handles batch encoding of text inputs and converts them into
    /// the tensor format expected by models.
    ///
    /// # Example  
    /// ```ignore
    /// use candle_utils::tokenization::encode_tokens;
    ///
    /// let device = candle_utils::get_device(true, false).unwrap();
    /// let inputs = ["Hello world", "How are you?"];
    /// let (tokens, token_ids, token_type_ids) = encode_tokens(&inputs, &tokenizer, &device).unwrap();
    /// ```
    pub fn encode_tokens(
        inputs: &[&str],
        tokenizer: &Tokenizer,
        device: &Device,
    ) -> Result<(Vec<Encoding>, Tensor, Tensor)> {
        let tokens = tokenizer
            .encode_batch(inputs.to_owned(), true)
            .map_err(|e| candle::Error::Msg(format!("Tokenizer error: {}", e)))?;

        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), device)
            })
            .collect::<Result<Vec<_>>>()?;
        let token_ids = Tensor::stack(&token_ids, 0)?;

        let token_type_ids = Tensor::zeros(token_ids.dims(), token_ids.dtype(), device)?;

        Ok((tokens, token_ids, token_type_ids))
    }

    /// Builds attention mask tensor from tokenizer encodings.
    ///
    /// The attention mask indicates which tokens should be attended to (1) and
    /// which should be ignored (0), typically used to mask padding tokens.
    ///
    /// # Example  
    /// ```ignore
    /// use candle_utils::tokenization::build_attention_mask;
    ///
    /// let device = candle_utils::get_device(true, false).unwrap();
    /// let attention_mask = build_attention_mask(&tokens, &device).unwrap();
    /// ```
    pub fn build_attention_mask(tokens: &[Encoding], device: &Device) -> Result<Tensor> {
        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Tensor::new(tokens.as_slice(), device)
            })
            .collect::<Result<Vec<_>>>()?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;

        Ok(attention_mask)
    }
}

/// Model and configuration loading utilities.
pub mod loader {
    use std::path::{Path, PathBuf};

    use candle::{Device, Result};
    use candle_nn::VarBuilder;
    use candle_transformers::models::{LoadableModel, ModelConfig};
    use hf_hub::{
        api::sync::{Api, ApiRepo},
        Repo, RepoType,
    };
    use tokenizers::Tokenizer;

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
        weights_files: Option<&Path>,
    ) -> Result<(Tokenizer, C, M)>
    where
        M: LoadableModel<C>,
        C: ModelConfig,
    {
        let repo = load_repo(model_id, revision)?;
        let tokenizer = load_tokenizer(&repo, tokenizer_file)?;
        let config = load_config(&repo, config_file)?;
        let model = load_model(&repo, device, dtype, &config, weights_files)?;

        Ok((tokenizer, config, model))
    }

    /// Creates a HuggingFace API repository handle.
    pub fn load_repo(model_id: &str, revision: &str) -> Result<ApiRepo> {
        let api = Api::new()
            .map_err(|e| candle::Error::Msg(format!("Couldn't create API. Error: {}", e)))?;

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
            None => repo.get("tokenizer.json").map_err(|e| {
                candle::Error::Msg(format!("Couldn't load tokenizer.json. Error: {}", e))
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
            None => repo.get("config.json").map_err(|e| {
                candle::Error::Msg(format!("Couldn't load config.json. Error: {}", e))
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
        weights_files: Option<&Path>,
    ) -> Result<M>
    where
        M: LoadableModel<C>,
        C: ModelConfig,
    {
        let weights_filename = match weights_files {
            Some(files) => PathBuf::from(files),
            None => {
                // Try safetensors first, then pytorch
                repo.get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .map_err(|e| candle::Error::Msg(format!("Model weights not found. Expected 'model.safetensors' or 'pytorch_model.bin'. Error: {}", e)))?
            }
        };

        let vb = if weights_filename.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, device)? }
        } else {
            println!("Loading weights from pytorch_model.bin");
            VarBuilder::from_pth(&weights_filename, dtype, device)?
        };

        M::load(vb, config)
    }
}
