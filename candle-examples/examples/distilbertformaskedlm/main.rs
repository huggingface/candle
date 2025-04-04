#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use std::ops::Sub;

use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::distilbert::{Config, DTYPE};
use candle_transformers::models::distilbertformaskedlm::DistilBertForMaskedLM;

use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model to use, check out available models on HuggingFace
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// When set, perform masked language modeling on this prompt.
    #[arg(long)]
    prompt: String,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long)]
    use_pth: bool,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// Whether to show the top-k predictions for masked tokens
    #[arg(long, default_value = "5")]
    top_k: usize,
}

impl Args {
    fn build_model_and_tokenizer(&self) -> Result<(DistilBertForMaskedLM, Tokenizer)> {
        let device = candle_examples::device(self.cpu)?;
        let default_model = "distilbert-base-uncased".to_string();
        let default_revision = "main".to_string();
        let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if self.use_pth {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = if self.use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        let model = DistilBertForMaskedLM::load(vb, &config)?;
        Ok((model, tokenizer))
    }
}

fn get_attention_mask(size: usize, device: &Device) -> Tensor {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device).unwrap()
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
    let device = &model.distilbert.device;

    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    let tokens = tokenizer
        .encode(args.prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let attention_mask = get_attention_mask(tokens.len(), device);

    println!("token_ids: {:?}", token_ids.to_vec2::<u32>());
    println!("mask: {:?}", attention_mask.to_vec2::<u8>());


    let mask_token_id = tokenizer
        .token_to_id("[MASK]")
        .ok_or(E::msg("No mask token found"))?;
    let masked_positions: Vec<u32> = tokens
        .iter()
        .enumerate()
        .filter_map(|(idx, &id)| if id == mask_token_id { Some(idx as u32) } else { None })
        .collect();

    if masked_positions.is_empty() {
        println!("No [MASK] tokens found after tokenization.");
        return Ok(());
    }

    let masked_lm_positions = Tensor::from_vec(masked_positions.clone(), (masked_positions.len(),), device)?;

    let logits = model.forward(&token_ids, &attention_mask, Some(&masked_lm_positions))?;
    println!("{}", logits);

    let mut logits_processor = LogitsProcessor::from_sampling(
        299792458,
        Sampling::TopK {
            k: 10,
            temperature: 0.8,
        },
    );

    for &pos in &masked_positions {
        println!("{:?}", logits.shape());
        let token_logits = logits.get(0)?.get(pos.sub(2) as usize)?;
        let probs = candle_nn::ops::softmax(&token_logits, 0)?;
        let tok_probs = probs.to_vec1::<f32>()?;
        let next_token_id = logits_processor.sample(&token_logits)?;
        let prob = tok_probs[next_token_id as usize];
        let token = tokenizer
            .id_to_token(next_token_id)
            .unwrap_or_else(|| format!("UNKNOWN_{}", next_token_id));
        println!(
            "Predictions for masked token at position {}: token={} prob={}",
            prob, token, prob
        );
    }

    Ok(())
}
