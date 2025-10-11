use std::path::PathBuf;

use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert;
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

#[derive(Debug, Clone, ValueEnum)]
enum Model {
    ModernBertBase,
    ModernBertLarge,
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

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long, default_value = "modern-bert-base")]
    model: Model,

    // Path to the tokenizer file.
    #[arg(long)]
    tokenizer_file: Option<String>,

    // Path to the weight files.
    #[arg(long)]
    weight_files: Option<String>,

    // Path to the config file.
    #[arg(long)]
    config_file: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let api = Api::new()?;
    let model_id = match &args.model_id {
        Some(model_id) => model_id.to_string(),
        None => match args.model {
            Model::ModernBertBase => "answerdotai/ModernBERT-base".to_string(),
            Model::ModernBertLarge => "answerdotai/ModernBERT-large".to_string(),
        },
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));

    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };

    let config_filename = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("config.json")?,
    };

    let weights_filename = match args.weight_files {
        Some(files) => PathBuf::from(files),
        None => match repo.get("model.safetensors") {
            Ok(safetensors) => safetensors,
            Err(_) => match repo.get("pytorch_model.bin") {
                Ok(pytorch_model) => pytorch_model,
                Err(e) => {
                    anyhow::bail!("Model weights not found. The weights should either be a `model.safetensors` or `pytorch_model.bin` file.  Error: {e}")
                }
            },
        },
    };

    let config = std::fs::read_to_string(config_filename)?;
    let config: modernbert::Config = serde_json::from_str(&config)?;
    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let device = candle_examples::device(args.cpu)?;

    let vb = if weights_filename.ends_with("model.safetensors") {
        unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], candle::DType::F32, &device)
                .unwrap()
        }
    } else {
        println!("Loading weights from pytorch_model.bin");
        VarBuilder::from_pth(&weights_filename, candle::DType::F32, &device).unwrap()
    };
    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_id: config.pad_token_id,
            ..Default::default()
        }))
        .with_truncation(None)
        .map_err(E::msg)?;

    let prompt = match &args.prompt {
        Some(p) => vec![p.as_str()],
        None => vec![
            "Hello I'm a [MASK] model.",
            "I'm a [MASK] boy.",
            "I'm [MASK] in berlin.",
            "The capital of France is [MASK].",
        ],
    };
    let model = modernbert::ModernBertForMaskedLM::load(vb, &config)?;

    let input_ids = tokenize_batch(&tokenizer, prompt.clone(), &device)?;
    let attention_mask = get_attention_mask(&tokenizer, prompt.clone(), &device)?;

    let output = model
        .forward(&input_ids, &attention_mask)?
        .to_dtype(candle::DType::F32)?;

    let max_outs = output.argmax(2)?;

    let max_out = max_outs.to_vec2::<u32>()?;
    let max_out_refs: Vec<&[u32]> = max_out.iter().map(|v| v.as_slice()).collect();
    let decoded = tokenizer.decode_batch(&max_out_refs, true).unwrap();
    for (i, sentence) in decoded.iter().enumerate() {
        println!("Sentence: {} : {}", i + 1, sentence);
    }

    Ok(())
}

pub fn tokenize_batch(
    tokenizer: &Tokenizer,
    input: Vec<&str>,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = tokenizer.encode_batch(input, true).map_err(E::msg)?;

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle::Result<Vec<_>>>()?;

    Ok(Tensor::stack(&token_ids, 0)?)
}

pub fn get_attention_mask(
    tokenizer: &Tokenizer,
    input: Vec<&str>,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = tokenizer.encode_batch(input, true).map_err(E::msg)?;

    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle::Result<Vec<_>>>()?;
    Ok(Tensor::stack(&attention_mask, 0)?)
}
