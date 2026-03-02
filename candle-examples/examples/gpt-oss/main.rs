#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::gpt_oss::{Config, ModelForCausalLM};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about = "Quick GPT-OSS inference (Rust/Candle)")]
struct Args {
    /// Use CPU instead of GPU.
    #[arg(long)]
    cpu: bool,

    /// HF model id to download from when --model-dir is not set.
    #[arg(long, default_value = "openai/gpt-oss-20b")]
    model_id: String,

    /// HF model revision.
    #[arg(long, default_value = "main")]
    revision: String,

    /// Local checkpoint directory containing config.json and model.safetensors.index.json.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Optional tokenizer.json path.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Prompt text.
    #[arg(long, default_value = "Write a short Rust function that reverses a UTF-8 string.")]
    prompt: String,

    /// Number of tokens to generate.
    #[arg(long, default_value_t = 128)]
    sample_len: usize,

    /// Temperature (0 => greedy).
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// Top-p sampling cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Top-k sampling cutoff.
    #[arg(long)]
    top_k: Option<usize>,

    /// RNG seed.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Repetition penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// Context size for repetition penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Process prompt token-by-token rather than one prefill pass.
    #[arg(long)]
    split_prompt: bool,

    /// Model dtype: auto, bf16, f16, f32.
    #[arg(long, default_value = "auto")]
    dtype: String,
}

fn resolve_dtype(args: &Args, device: &Device) -> Result<DType> {
    let dtype = match args.dtype.as_str() {
        "auto" => {
            if device.is_cuda() || device.is_metal() {
                DType::BF16
            } else {
                DType::F32
            }
        }
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        "f32" => DType::F32,
        other => anyhow::bail!("unsupported --dtype {other}, expected auto|bf16|f16|f32"),
    };
    Ok(dtype)
}

fn load_paths(args: &Args) -> Result<(Vec<PathBuf>, PathBuf, PathBuf)> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        args.revision.clone(),
    ));

    match &args.model_dir {
        Some(model_dir) => {
            let model_files =
                candle_examples::hub_load_local_safetensors(model_dir, "model.safetensors.index.json")?;
            let config_file = model_dir.join("config.json");
            let tokenizer_file = match &args.tokenizer {
                Some(p) => p.clone(),
                None => {
                    let local = model_dir.join("tokenizer.json");
                    if local.exists() {
                        local
                    } else {
                        repo.get("tokenizer.json")?
                    }
                }
            };
            Ok((model_files, config_file, tokenizer_file))
        }
        None => {
            let model_files = candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;
            let config_file = repo.get("config.json")?;
            let tokenizer_file = match &args.tokenizer {
                Some(p) => p.clone(),
                None => repo.get("tokenizer.json")?,
            };
            Ok((model_files, config_file, tokenizer_file))
        }
    }
}

fn eos_token_id(tos: &TokenOutputStream) -> Option<u32> {
    tos.get_token("<|endoftext|>")
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let dtype = resolve_dtype(&args, &device)?;

    let (model_files, config_file, tokenizer_file) = load_paths(&args)?;
    println!("loading config from {}", config_file.display());
    println!("loading tokenizer from {}", tokenizer_file.display());
    println!(
        "loading {} model shard(s), dtype={dtype:?}, device={:?}",
        model_files.len(),
        device
    );

    let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(E::msg)?;
    let mut tos = TokenOutputStream::new(tokenizer);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)? };
    let mut model = ModelForCausalLM::new(&config, vb)?;

    let tokens = tos
        .tokenizer()
        .encode(args.prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    if tokens.is_empty() {
        anyhow::bail!("tokenizer produced no tokens for prompt");
    }
    println!("prompt tokens: {}", tokens.len());

    let sampling = if args.temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        match (args.top_k, args.top_p) {
            (None, None) => Sampling::All {
                temperature: args.temperature,
            },
            (Some(k), None) => Sampling::TopK {
                k,
                temperature: args.temperature,
            },
            (None, Some(p)) => Sampling::TopP {
                p,
                temperature: args.temperature,
            },
            (Some(k), Some(p)) => Sampling::TopKThenTopP {
                k,
                p,
                temperature: args.temperature,
            },
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling);

    let start_prefill = std::time::Instant::now();
    let mut next_token = if !args.split_prompt {
        let input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        logits_processor.sample(&logits)?
    } else {
        let mut next = 0u32;
        for (pos, &tok) in tokens.iter().enumerate() {
            let input = Tensor::new(&[tok], &device)?.unsqueeze(0)?;
            let logits = model
                .forward(&input, pos)?
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;
            next = logits_processor.sample(&logits)?;
        }
        next
    };
    let prefill_dt = start_prefill.elapsed();

    print!("{}", args.prompt);
    std::io::stdout().flush()?;
    let mut generated_tokens = vec![next_token];
    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        std::io::stdout().flush()?;
    }

    let eos = eos_token_id(&tos);
    let start_gen = std::time::Instant::now();
    let mut sampled = 1usize;
    for idx in 0..args.sample_len.saturating_sub(1) {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model
            .forward(&input, tokens.len() + idx)?
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;

        let logits = if args.repeat_penalty == 1.0 {
            logits
        } else {
            let start_at = generated_tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &generated_tokens[start_at..],
            )?
        };

        next_token = logits_processor.sample(&logits)?;
        generated_tokens.push(next_token);
        sampled += 1;

        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        if let Some(eos) = eos {
            if next_token == eos {
                break;
            }
        }
    }

    if let Some(rest) = tos.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    println!();

    let gen_dt = start_gen.elapsed();
    println!(
        "stats: prompt={} tok ({:.2} tok/s), generated={} tok ({:.2} tok/s)",
        tokens.len(),
        tokens.len() as f64 / prefill_dt.as_secs_f64(),
        sampled,
        sampled as f64 / gen_dt.as_secs_f64()
    );

    Ok(())
}
