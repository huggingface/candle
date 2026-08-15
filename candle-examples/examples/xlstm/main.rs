//! xLSTM text generation example
//!
//! This example demonstrates text generation using the xLSTM model from NX-AI.
//!
//! ```bash
//! # Run with default settings (requires ~28GB for 7B model weights)
//! cargo run --example xlstm --release --features cuda -- --prompt "Once upon a time"
//!
//! # Run on CPU with bf16 or f32
//! cargo run --example xlstm --release -- --cpu --prompt "Hello, world" --dtype f32
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::xlstm::{Config, Model};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        bos_token_id: u32,
        eos_token_id: u32,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            bos_token_id,
            eos_token_id,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let dtype = self.model.dtype();

        // Encode prompt tokens
        let prompt_tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        // Prepend BOS token as required by xLSTM (force_bos_token_insert: true)
        let mut tokens = vec![self.bos_token_id];
        tokens.extend(prompt_tokens);

        let mut generated_tokens = 0usize;
        let mut state = self.model.new_state(1, &self.device)?;
        let mut next_logits = None;

        // Process BOS + prompt tokens
        println!("\nTokens (with BOS): {:?}", tokens);
        for (i, &t) in tokens.iter().enumerate() {
            let input = Tensor::new(&[t], &self.device)?;
            let logits = self.model.forward(&input, &mut state)?;
            next_logits = Some(logits);

            // Print token (skip BOS for display)
            if i > 0 {
                if let Some(text) = self.tokenizer.next_token(t)? {
                    print!("{text}")
                }
            }
        }
        std::io::stdout().flush()?;

        let start_gen = std::time::Instant::now();

        // Generate new tokens
        for _ in 0..sample_len {
            let logits = match next_logits.as_ref() {
                Some(logits) => logits,
                None => anyhow::bail!("cannot work on an empty prompt"),
            };
            let logits = logits.squeeze(0)?.to_dtype(dtype)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;

            if next_token == self.eos_token_id {
                break;
            }

            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }

            let input = Tensor::new(&[next_token], &self.device)?;
            next_logits = Some(self.model.forward(&input, &mut state)?)
        }

        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "xLSTM text generation", long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The prompt to generate from.
    #[arg(long, default_value = "Once upon a time")]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 100)]
    sample_len: usize,

    /// HuggingFace model ID.
    #[arg(long, default_value = "NX-AI/xLSTM-7b")]
    model_id: String,

    /// Model revision.
    #[arg(long, default_value = "main")]
    revision: String,

    /// Path to local tokenizer file.
    #[arg(long)]
    tokenizer_file: Option<String>,

    /// Comma-separated paths to local weight files.
    #[arg(long)]
    weight_files: Option<String>,

    /// Path to local config file.
    #[arg(long)]
    config_file: Option<String>,

    /// Data type: f32, bf16, or f16.
    #[arg(long, default_value = "bf16")]
    dtype: String,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    use std::str::FromStr;
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    // Warn if top_p is set without temperature (top_p requires temperature to work)
    if args.top_p.is_some() && args.temperature.is_none() {
        eprintln!(
            "Warning: --top-p has no effect without --temperature. \
             Using greedy sampling. Try: --temperature 0.7 --top-p 0.9"
        );
    }

    // Warn if seed is set without temperature (seed only affects random sampling)
    if args.temperature.is_none() {
        eprintln!(
            "Warning: --seed has no effect without --temperature. \
             Greedy sampling is deterministic. Try: --temperature 0.7 --seed {}",
            args.seed
        );
    }

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        args.revision.clone(),
    ));

    // Get tokenizer
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };

    // Get config
    let config_filename = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("config.json")?,
    };

    // Get weight files (xLSTM-7b has 6 shards)
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            // xLSTM-7b uses 6 safetensor shards
            let mut files = Vec::new();
            for i in 1..=6 {
                let filename = format!("model-{i:05}-of-00006.safetensors");
                files.push(repo.get(&filename)?);
            }
            files
        }
    };

    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    println!(
        "Config: vocab_size={}, embedding_dim={}, num_blocks={}, num_heads={}",
        config.vocab_size, config.embedding_dim, config.num_blocks, config.num_heads
    );

    let device = candle_examples::device(args.cpu)?;
    let dtype = DType::from_str(&args.dtype)?;
    println!("Loading model on {:?} with dtype {:?}", device, dtype);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = Model::new(&config, vb)?;
    println!("loaded the model in {:?}", start.elapsed());

    // Use tokenizer's special tokens (config.json has wrong eos_token_id=2 which is '!')
    let bos_token_id = tokenizer
        .token_to_id("<|endoftext|>")
        .unwrap_or(config.bos_token_id as u32);
    let eos_token_id = bos_token_id; // For this model, BOS and EOS are the same token

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        bos_token_id,
        eos_token_id,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
