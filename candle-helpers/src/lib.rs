//! Helper utils for common Candle use cases.

use std::path::{Path, PathBuf};

use candle::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{LoadableModel, ModelConfig};
use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};
use tokenizers::{Encoding, Tokenizer};

/// Helper function to get a single candle Device. This will not perform any multi device mapping.
/// Will prioritize Metal or CUDA (if Candle is compiled with those features) and fallback to CPU.
///
/// ## Example  
/// ```
/// let device = candle_core::utils::device(true, false).unwrap();
/// ```
pub fn device(use_cpu: bool, quiet: bool) -> Result<Device> {
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
                println!("Running on CPU, to run on GPU (metal), build with `--features metal`");
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                println!("Running on CPU, to run on GPU, build with `--features cuda`");
            }
        }

        Ok(Device::Cpu)
    }
}

/// Perform L2 normalization.
///
/// ## Example  
/// ```
/// use candle_core::{Tensor, utils::normalize_l2};
///
/// let device = candle_core::utils::device(true, false).unwrap();
/// let x = Tensor::new(&[[0f32, 1.], [2., 3.]], &device).unwrap();
/// let normalized_x = normalize_l2(&x).unwrap();
/// ```
pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    let squared = &v.sqr()?;
    let summed = &squared.sum_keepdim(1)?;
    let norms = &summed.sqrt()?;
    v.broadcast_div(norms)
}

/// Encodes the input strings into tokens, IDs, and type IDs.
///
/// ## Example  
/// ```ignore
/// use candle_helpers::encode_tokens;
///
/// let device = candle_core::utils::device(true, false).unwrap();
/// let (tokens, token_ids, token_type_ids) = encode_tokens(inputs, &tokenizer, &device).unwrap();
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

/// Builds the attention mask from the token encodings.
///
/// ## Example  
/// ```ignore
/// use candle_helpers::build_attention_mask;
///
/// let device = candle_core::utils::device(true, false).unwrap();
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

/// Loads the tokenizer config and model.
///
/// ## Example  
/// ```ignore
/// use candle_helpers::load_model_and_tokenizer;
/// use candle_transformers::models::modernbert::{Config, ModernBertForMaskedLM};
///
/// let (config, model, mut tokenizer) = load_tokenizer_config_model::<ModernBertForMaskedLM, Config>(
///         &device,
///         &model_id,
///         &args.revision,
///         candle::DType::F32,
///         None,
///         None,
///         None,
///     )?;
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

pub fn load_repo(model_id: &str, revision: &str) -> Result<ApiRepo> {
    let api =
        Api::new().map_err(|e| candle::Error::Msg(format!("Couldn't create API. Error: {}", e)))?;

    Ok(api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    )))
}

pub fn load_tokenizer(repo: &ApiRepo, tokenizer_file: Option<&Path>) -> Result<Tokenizer> {
    let tokenizer_filepath = match tokenizer_file {
        Some(file) => PathBuf::from(file),
        None => repo.get("tokenizer.json").map_err(|e| {
            candle::Error::Msg(format!("Couldn't load tokenizers.json. Error: {}", e))
        })?,
    };

    Tokenizer::from_file(tokenizer_filepath)
        .map_err(|e| candle::Error::Msg(format!("Couldn't load tokenizers. Error: {}", e)))
}

pub fn load_config<C>(repo: &ApiRepo, config_file: Option<&Path>) -> Result<C>
where
    C: ModelConfig,
{
    let config_filepath = match config_file {
        Some(file) => PathBuf::from(file),
        None => repo
            .get("config.json")
            .map_err(|e| candle::Error::Msg(format!("Couldn't load config.json. Error: {}", e)))?,
    };

    let config_str = std::fs::read_to_string(config_filepath)?;

    let config: C = serde_json::from_str(&config_str)
        .map_err(|e| candle::Error::Msg(format!("Couldn't load config. Error: {}", e)))?;

    Ok(config)
}

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
