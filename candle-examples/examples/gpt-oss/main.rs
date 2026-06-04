//! gpt-oss (OpenAI open-weights MoE) text generation.
//!
//! cargo run --release --example gpt-oss -- --prompt "The capital of France is"
//! (add --features cuda for GPU; bf16 is used on CUDA, f32 on CPU)
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
use candle_transformers::models::gpt_oss::{Config, ModelForCausalLM};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

struct TextGeneration {
    model: ModelForCausalLM,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: ModelForCausalLM,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.model.clear_kv_cache();
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        // Harmony stop tokens; keep whichever the tokenizer defines.
        let eos: Vec<u32> = ["<|return|>", "<|endoftext|>", "<|call|>"]
            .iter()
            .filter_map(|t| self.tokenizer.get_token(t))
            .collect();

        let mut generated = 0usize;
        let start = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
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
            let next = self.logits_processor.sample(&logits)?;
            tokens.push(next);
            generated += 1;
            if eos.contains(&next) {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated} tokens generated ({:.2} token/s)",
            generated as f64 / dt.as_secs_f64()
        );
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than the GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value = "The capital of France is")]
    prompt: String,

    #[arg(long, default_value_t = 256)]
    sample_len: usize,

    #[arg(long, default_value = "openai/gpt-oss-20b")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    temperature: Option<f64>,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let config_filename = repo.get("config.json")?;
    let filenames = candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;

    let start = std::time::Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = ModelForCausalLM::new(&config, vb)?;
    println!("loaded model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    println!("prompt: {:?}", args.prompt);
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
