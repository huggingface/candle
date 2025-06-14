use std::path::Path;

use anyhow::{Error as E, Result};

use candle_helpers::{build_attention_mask, device, encode_tokens, load_tokenizer_config_model};
use candle_transformers::models::modernbert::{Config, ModernBertForMaskedLM};
use clap::{Parser, ValueEnum};
use tokenizers::PaddingParams;

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
    let device = device(args.cpu, false)?;

    let model_id = match &args.model_id {
        Some(model_id) => model_id.to_string(),
        None => match args.model {
            Model::ModernBertBase => "answerdotai/ModernBERT-base".to_string(),
            Model::ModernBertLarge => "answerdotai/ModernBERT-large".to_string(),
        },
    };

    let tokenizer_file = args.tokenizer_file.as_deref().map(Path::new);
    let config_file = args.config_file.as_deref().map(Path::new);
    let weights_files = args.weight_files.as_deref().map(Path::new);

    let (mut tokenizer, config, model) =
        load_tokenizer_config_model::<ModernBertForMaskedLM, Config>(
            &device,
            &model_id,
            &args.revision,
            candle::DType::F32,
            tokenizer_file,
            config_file,
            weights_files,
        )?;

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

    let (tokens, input_ids, _token_type_ids) = encode_tokens(&prompt, &tokenizer, &device)?;
    let attention_mask = build_attention_mask(&tokens, &device)?;

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
