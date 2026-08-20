//! AutoModelForCausalLM example - load any supported model by name.
//!
//! ```bash
//! cargo run --example auto --release -- --model "Qwen/Qwen2-0.5B-Instruct" --prompt "Hello, world"
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::auto::{AutoModelForCausalLM, AutoModelOptions};
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // HuggingFace model ID
    #[arg(long)]
    model: String,

    #[arg(long, default_value = "Hello, I am")]
    prompt: String,

    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 100)]
    max_tokens: usize,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long, default_value_t = true, long_help = "Run on CPU by default otherwise on GPU")]
    cpu: bool,

    #[arg(long, long_help = "Use Flash Attention")]
    flash_attn: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading model: {}", args.model);

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    // Load tokenizer
    let api = Api::new()?;
    let revision = args.revision.as_deref().unwrap_or("main");
    let repo = api.repo(Repo::with_revision(
        args.model.clone(),
        RepoType::Model,
        revision.to_string(),
    ));
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    // Load model using AutoModelForCausalLM
    let options = AutoModelOptions {
        revision: args.revision.clone(),
        use_flash_attn: args.flash_attn,
    };
    let mut model = AutoModelForCausalLM::from_pretrained(&args.model, dtype, &device, options)?;

    println!("Model loaded: {}", model.model_type());
    println!("Generating from prompt: \"{}\"", args.prompt);
    println!();

    // Tokenize prompt
    let tokens = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let mut token_output = TokenOutputStream::new(tokenizer);
    let mut logits_processor = LogitsProcessor::new(
        args.seed,
        Some(args.temperature),
        args.top_p,
    );

    let mut all_tokens = tokens.clone();
    let mut generated = 0;

    // Generate tokens
    for index in 0..args.max_tokens {
        let context_size = if index == 0 { tokens.len() } else { 1 };
        let start_pos = all_tokens.len().saturating_sub(context_size);
        let input = Tensor::new(&all_tokens[start_pos..], &device)?.unsqueeze(0)?;

        let logits = model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;

        // Apply repeat penalty
        let logits = if args.repeat_penalty != 1.0 {
            let start = all_tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &all_tokens[start..],
            )?
        } else {
            logits
        };

        let next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        generated += 1;

        // Print token
        if let Some(text) = token_output.next_token(next_token)? {
            print!("{text}");
            std::io::Write::flush(&mut std::io::stdout())?;
        }

        // Stop on EOS
        if let Some(eos) = token_output.tokenizer().get_vocab(true).get("<|endoftext|>") {
            if next_token == *eos {
                break;
            }
        }
        if let Some(eos) = token_output.tokenizer().get_vocab(true).get("</s>") {
            if next_token == *eos {
                break;
            }
        }
    }

    // Flush remaining tokens
    if let Some(text) = token_output.decode_rest().map_err(E::msg)? {
        print!("{text}");
    }
    println!();
    println!("\n[Generated {} tokens]", generated);

    Ok(())
}

