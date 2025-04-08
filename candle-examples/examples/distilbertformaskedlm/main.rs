#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use std::ops::Sub;
use std::path::PathBuf;

use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::distilbert::{Config, DTYPE};
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
    let attention_mask = get_attention_mask(&tokenizer, prompt.clone(), &device)?;
    let attention_mask = attention_mask
        .unsqueeze(1)?  // Shape becomes [4, 1, 10]
        .unsqueeze(3)?;  // Shape becomes [4, 1, 10, 1]

    println!("{:?}", attention_mask.shape());

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


// #[derive(Parser, Debug)]
// #[command(author, version, about, long_about = None)]
// struct Args {
//     /// Run on CPU rather than on GPU.
//     #[arg(long)]
//     cpu: bool,

//     /// Enable tracing (generates a trace-timestamp.json file).
//     #[arg(long)]
//     tracing: bool,

//     /// The model to use, check out available models on HuggingFace
//     #[arg(long)]
//     model_id: Option<String>,

//     #[arg(long)]
//     revision: Option<String>,

//     /// When set, perform masked language modeling on this prompt.
//     #[arg(long)]
//     prompt: String,

//     /// Use the pytorch weights rather than the safetensors ones
//     #[arg(long)]
//     use_pth: bool,

//     /// The number of times to run the prompt.
//     #[arg(long, default_value = "1")]
//     n: usize,

//     /// Whether to show the top-k predictions for masked tokens
//     #[arg(long, default_value = "5")]
//     top_k: usize,
// }

// impl Args {
//     fn build_model_and_tokenizer(&self) -> Result<(DistilBertForMaskedLM, Tokenizer)> {
//         let device = candle_examples::device(self.cpu)?;
//         let default_model = "distilbert-base-uncased".to_string();
//         let default_revision = "main".to_string();
//         let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
//             (Some(model_id), Some(revision)) => (model_id, revision),
//             (Some(model_id), None) => (model_id, "main".to_string()),
//             (None, Some(revision)) => (default_model, revision),
//             (None, None) => (default_model, default_revision),
//         };

//         let repo = Repo::with_revision(model_id, RepoType::Model, revision);
//         let (config_filename, tokenizer_filename, weights_filename) = {
//             let api = Api::new()?;
//             let api = api.repo(repo);
//             let config = api.get("config.json")?;
//             let tokenizer = api.get("tokenizer.json")?;
//             let weights = if self.use_pth {
//                 api.get("pytorch_model.bin")?
//             } else {
//                 api.get("model.safetensors")?
//             };
//             (config, tokenizer, weights)
//         };
//         let config = std::fs::read_to_string(config_filename)?;
//         let config: Config = serde_json::from_str(&config)?;
//         let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

//         let vb = if self.use_pth {
//             VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
//         } else {
//             unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
//         };
//         let model = DistilBertForMaskedLM::load(vb, &config)?;
//         Ok((model, tokenizer))
//     }
// }

// fn get_attention_mask(size: usize, device: &Device) -> Tensor {
//     let mask: Vec<_> = (0..size)
//         .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
//         .collect();
//     Tensor::from_slice(&mask, (size, size), device).unwrap()
// }

// fn main() -> Result<()> {
//     use tracing_chrome::ChromeLayerBuilder;
//     use tracing_subscriber::prelude::*;

//     let args = Args::parse();
//     let _guard = if args.tracing {
//         println!("tracing...");
//         let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
//         tracing_subscriber::registry().with(chrome_layer).init();
//         Some(guard)
//     } else {
//         None
//     };
//     let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
//     let device = &model.bert.device;

//     let tokenizer = tokenizer
//         .with_padding(None)
//         .with_truncation(None)
//         .map_err(E::msg)?;
//     let tokens = tokenizer
//         .encode(args.prompt, true)
//         .map_err(E::msg)?
//         .get_ids()
//         .to_vec();
//     let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
//     let attention_mask = get_attention_mask(tokens.len(), device);

//     println!("token_ids: {:?}", token_ids.to_vec2::<u32>());
//     println!("mask: {:?}", attention_mask.to_vec2::<u8>());


//     let mask_token_id = tokenizer
//         .token_to_id("[MASK]")
//         .ok_or(E::msg("No mask token found"))?;
//     let masked_positions: Vec<u32> = tokens
//         .iter()
//         .enumerate()
//         .filter_map(|(idx, &id)| if id == mask_token_id { Some(idx as u32) } else { None })
//         .collect();

//     if masked_positions.is_empty() {
//         println!("No [MASK] tokens found after tokenization.");
//         return Ok(());
//     }

//     let masked_lm_positions = Tensor::from_vec(masked_positions.clone(), (masked_positions.len(),), device)?;

//     let logits = model.forward(&token_ids, &attention_mask)?;
//     println!("{}", logits);

//     let mut logits_processor = LogitsProcessor::from_sampling(
//         299792458,
//         Sampling::TopK {
//             k: 10,
//             temperature: 0.8,
//         },
//     );

//     for &pos in &masked_positions {
//         println!("{:?}", logits.shape());
//         let token_logits = logits.get(0)?.get(pos.sub(2) as usize)?;
//         let probs = candle_nn::ops::softmax(&token_logits, 0)?;
//         let tok_probs = probs.to_vec1::<f32>()?;
//         let next_token_id = logits_processor.sample(&token_logits)?;
//         let prob = tok_probs[next_token_id as usize];
//         let token = tokenizer
//             .id_to_token(next_token_id)
//             .unwrap_or_else(|| format!("UNKNOWN_{}", next_token_id));
//         println!(
//             "Predictions for masked token at position {}: token={} prob={}",
//             pos, token, prob
//         );
//     }

//     Ok(())
// }
