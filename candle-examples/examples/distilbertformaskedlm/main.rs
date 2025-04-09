#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use std::path::PathBuf;

use candle_transformers::models::distilbert::Config;
use candle_transformers::models::distilbert::DistilBertForMaskedLM;

use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

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
    let model_id = "distilbert/distilbert-base-uncased".to_string();
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
    let config: Config = serde_json::from_str(&config)?;
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
            pad_id: config.pad_token_id as u32,
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
    let model = DistilBertForMaskedLM::load(vb, &config)?;

    let input_ids = tokenize_batch(&tokenizer, prompt.clone(), &device)?;
    let attention_mask = get_proper_attention_mask(&tokenizer, prompt.clone(), &device)?;

    println!("{:?}", attention_mask.get(0)?.get(0)?.to_vec2::<u8>()?);
    let output = model
        .forward(&input_ids, &attention_mask)?
        .to_dtype(candle::DType::F32)?;

    let max_outs = output.argmax(2)?;
    let max_out = max_outs.to_vec2::<u32>()?;
    println!("Sample predicted token IDs: {:?}", &max_out[0][..10]); //All zeros

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

fn get_proper_attention_mask(
    tokenizer: &Tokenizer,
    input: Vec<&str>,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = tokenizer.encode_batch(input, true).map_err(E::msg)?;
    let batch_size = tokens.len();
    let seq_len = tokens.first().unwrap().get_attention_mask().to_vec().len();

    // Get the mask token ID
    let mask_token_id = tokenizer
        .token_to_id("[MASK]")
        .ok_or(anyhow::anyhow!("No mask token found"))?;

    let mut attention_mask_vec = Vec::with_capacity(batch_size * seq_len * seq_len);

    for token_encoding in tokens.iter() {
        let ids = token_encoding.get_ids();
        // let attention_mask = token_encoding.get_attention_mask();

        for _ in 0..seq_len {
            for id in ids.iter() {
                let mask_value = if id == &mask_token_id { 1u8 } else { 0u8 };
                attention_mask_vec.push(mask_value);
            }
        }
    }

    let shape = (batch_size, 1, seq_len, seq_len);
    let mask = Tensor::from_vec(attention_mask_vec, shape, device)?;

    Ok(mask)
}
