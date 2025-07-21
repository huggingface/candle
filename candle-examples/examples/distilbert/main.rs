#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use candle_transformers::models::distilbert::{
    Config, DistilBertForMaskedLM, DistilBertModel, DTYPE,
};

use anyhow::{Context, Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use tokenizers::Tokenizer;

enum ModelType {
    Masked(Box<DistilBertForMaskedLM>),
    UnMasked(Box<DistilBertModel>),
}

impl ModelType {
    fn device(&self) -> &Device {
        match self {
            ModelType::Masked(model) => &model.bert.device,
            ModelType::UnMasked(model) => &model.device,
        }
    }

    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        match self {
            ModelType::Masked(model) => Ok(model.forward(input_ids, attention_mask)?),
            ModelType::UnMasked(model) => Ok(model.forward(input_ids, attention_mask)?),
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    #[value(name = "distilbert")]
    DistilBert,

    #[value(name = "distilbertformaskedlm")]
    DistilbertForMaskedLM,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long, default_value = "distilbert")]
    model: Which,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model_id: Option<String>,

    /// Revision or branch
    #[arg(long)]
    revision: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: String,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long)]
    use_pth: bool,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// Number of top predictions to show for each mask
    #[arg(long, default_value = "5")]
    top_k: usize,
}

impl Args {
    fn build_model_and_tokenizer(&self) -> Result<(ModelType, Tokenizer)> {
        let device = candle_examples::device(self.cpu)?;

        let (model_id, revision) = self.resolve_model_and_revision();
        let (config_path, tokenizer_path, weights_path) =
            self.download_model_files(&model_id, &revision)?;

        let config = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

        let vb = self.load_variables(&weights_path, &device)?;
        let model = self.create_model(&config, vb)?;

        Ok((model, tokenizer))
    }

    fn resolve_model_and_revision(&self) -> (String, String) {
        let default_model = "distilbert-base-uncased".to_string();
        let default_revision = "main".to_string();

        match (self.model_id.clone(), self.revision.clone()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, default_revision),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        }
    }

    fn download_model_files(
        &self,
        model_id: &str,
        revision: &str,
    ) -> Result<(PathBuf, PathBuf, PathBuf)> {
        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
        let api = Api::new()?;
        let api = api.repo(repo);

        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = if self.use_pth {
            api.get("pytorch_model.bin")?
        } else {
            api.get("model.safetensors")?
        };

        Ok((config, tokenizer, weights))
    }

    fn load_variables(&self, weights_path: &PathBuf, device: &Device) -> Result<VarBuilder> {
        if self.use_pth {
            Ok(VarBuilder::from_pth(weights_path, DTYPE, device)?)
        } else {
            Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, device)? })
        }
    }

    fn create_model(&self, config: &Config, vb: VarBuilder) -> Result<ModelType> {
        match self.model {
            Which::DistilbertForMaskedLM => Ok(ModelType::Masked(
                DistilBertForMaskedLM::load(vb, config)?.into(),
            )),
            Which::DistilBert => Ok(ModelType::UnMasked(
                DistilBertModel::load(vb, config)?.into(),
            )),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _guard = setup_tracing(&args);

    let (model, tokenizer) = args.build_model_and_tokenizer()?;
    let device = model.device();

    let (token_ids, mask) = prepare_inputs(&args, &tokenizer, device)?;
    let output = model.forward(&token_ids, &mask)?;

    process_output(&model, &output, &token_ids, &tokenizer, &args)?;

    Ok(())
}

fn setup_tracing(args: &Args) -> Option<impl Drop> {
    if args.tracing {
        use tracing_chrome::ChromeLayerBuilder;
        use tracing_subscriber::prelude::*;

        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    }
}

fn prepare_inputs(args: &Args, tokenizer: &Tokenizer, device: &Device) -> Result<(Tensor, Tensor)> {
    let mut binding = tokenizer.clone();
    let tokenizer_configured = binding
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;

    let tokens = tokenizer_configured
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;

    let mask = match args.model {
        Which::DistilbertForMaskedLM => attention_mask_maskedlm(tokenizer, &args.prompt, device)?,
        Which::DistilBert => attention_mask(tokens.len(), device)?,
    };

    println!("token_ids: {:?}", token_ids.to_vec2::<u32>()?);

    Ok((token_ids, mask))
}

fn process_output(
    model: &ModelType,
    output: &Tensor,
    token_ids: &Tensor,
    tokenizer: &Tokenizer,
    args: &Args,
) -> Result<()> {
    match model {
        ModelType::UnMasked(_) => {
            println!("embeddings");
            println!("{output}");
        }
        ModelType::Masked(_) => {
            process_masked_output(output, token_ids, tokenizer, args)?;
        }
    }

    Ok(())
}

fn process_masked_output(
    output: &Tensor,
    token_ids: &Tensor,
    tokenizer: &Tokenizer,
    args: &Args,
) -> Result<()> {
    let input_ids_vec = token_ids.to_vec2::<u32>()?;
    let mask_token_id = tokenizer
        .token_to_id("[MASK]")
        .context("Mask token, \"[MASK]\", not found in tokenizer.")?;

    println!("\nInput: {}", args.prompt);

    for (token_idx, &token_id) in input_ids_vec[0].iter().enumerate() {
        if token_id == mask_token_id {
            println!("Predictions for [MASK] at position {token_idx}:");

            let pos_logits = output.get(0)?.get(token_idx)?;
            let probs = candle_nn::ops::softmax(&pos_logits, 0)?;
            let (top_values, top_indices) = get_top_k(&probs, args.top_k)?;

            let values = top_values.to_vec1::<f32>()?;
            let indices = top_indices.to_vec1::<u32>()?;

            for (i, (&token_id, &prob)) in indices.iter().zip(values.iter()).enumerate() {
                let token = tokenizer.decode(&[token_id], false).map_err(E::msg)?;
                println!(
                    "  {}: {:15} (probability: {:.2}%)",
                    i + 1,
                    token,
                    prob * 100.0
                );
            }
        }
    }

    Ok(())
}

fn get_top_k(tensor: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let n = tensor.dims().iter().product::<usize>();
    let k = std::cmp::min(k, n);

    let values = tensor.to_vec1::<f32>()?;
    let mut value_indices: Vec<(f32, usize)> = values
        .into_iter()
        .enumerate()
        .map(|(idx, val)| (val, idx))
        .collect();

    value_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let top_k_values: Vec<f32> = value_indices.iter().take(k).map(|(val, _)| *val).collect();
    let top_k_indices: Vec<u32> = value_indices
        .iter()
        .take(k)
        .map(|(_, idx)| *idx as u32)
        .collect();

    let device = tensor.device();
    let top_values = Tensor::from_vec(top_k_values, (k,), device)?;
    let top_indices = Tensor::from_vec(top_k_indices, (k,), device)?;

    Ok((top_values, top_indices))
}

fn attention_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Ok(Tensor::from_slice(&mask, (size, size), device)?)
}

fn attention_mask_maskedlm(tokenizer: &Tokenizer, input: &str, device: &Device) -> Result<Tensor> {
    let tokens = tokenizer.encode(input, true).map_err(E::msg)?;
    let seq_len = tokens.get_attention_mask().to_vec().len();

    let mask_token_id = tokenizer
        .token_to_id("[MASK]")
        .context("Mask token, \"[MASK]\", not found in tokenizer.")?;

    let mut attention_mask_vec = Vec::with_capacity(seq_len * seq_len);

    let ids = tokens.get_ids();
    for _ in 0..seq_len {
        for id in ids.iter() {
            let mask_value = if id == &mask_token_id { 1u8 } else { 0u8 };
            attention_mask_vec.push(mask_value);
        }
    }

    let shape = (1, 1, seq_len, seq_len);
    let mask = Tensor::from_vec(attention_mask_vec, shape, device)?;

    Ok(mask)
}
