// Granite 4.0 Micro text generation example (GraniteMoeHybrid).

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl-unlinked")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};
use clap::Parser;

use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::granitemoehybrid as model;
use hf_hub::{api::sync::Api, Repo, RepoType};
use model::{GraniteMoeHybrid, GraniteMoeHybridCache, GraniteMoeHybridConfig};

use std::{io::Write, path::Path};

use std::time::Instant;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

const EOS_TOKEN_ID: u32 = 100257;
const DEFAULT_PROMPT: &str = "How Fault Tolerant Quantum Computers will help humanity?";
const DEFAULT_MODEL_ID: &str = "ibm-granite/granite-4.0-micro";

fn build_chat_prompt(user_prompt: &str) -> String {
    format!(
        "<|start_of_role|>user<|end_of_role|>{user_prompt}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
    )
}

fn init_tracing(enable: bool) {
    if !enable {
        return;
    }
    let (chrome_layer, _) = ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 4096)]
    sample_len: usize,

    #[arg(long)]
    no_kv_cache: bool,

    #[arg(long)]
    prompt: Option<String>,

    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Override the model identifier or directory.
    #[arg(long)]
    model_id: Option<String>,

    /// Use a specific revision when loading from the Hugging Face Hub.
    #[arg(long)]
    revision: Option<String>,

    /// Enable Flash-Attention kernels when compiled with the feature.
    #[arg(long)]
    use_flash_attn: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    use candle_examples::token_output_stream::TokenOutputStream;
    use tokenizers::Tokenizer;

    let args = Args::parse();
    init_tracing(args.tracing);

    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => {
            if device.is_cuda() || device.is_metal() {
                DType::BF16
            } else {
                DType::F32
            }
        }
    };

    let (granite, tokenizer_filename, mut cache, config) = {
        let model_id = args
            .model_id
            .clone()
            .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());
        println!("Loading the model weights from {model_id}");

        if Path::new(&model_id).exists() {
            let model_path = Path::new(&model_id);
            let tokenizer_filename = model_path.join("tokenizer.json");
            let config_filename = model_path.join("config.json");
            let config: GraniteMoeHybridConfig =
                serde_json::from_slice(&std::fs::read(&config_filename)?)?;
            let config = config.into_config(args.use_flash_attn);
            let filenames = candle_examples::hub_load_local_safetensors(
                model_path,
                "model.safetensors.index.json",
            )?;
            let cache = GraniteMoeHybridCache::new(!args.no_kv_cache, dtype, &config, &device)?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            (
                GraniteMoeHybrid::load(vb, &config)?,
                tokenizer_filename,
                cache,
                config,
            )
        } else {
            let api = Api::new()?;
            let revision = args.revision.clone().unwrap_or_else(|| "main".to_string());
            let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

            let tokenizer_filename = repo.get("tokenizer.json")?;
            let config_filename = repo.get("config.json")?;
            let config: GraniteMoeHybridConfig =
                serde_json::from_slice(&std::fs::read(config_filename)?)?;
            let config = config.into_config(args.use_flash_attn);
            let filenames =
                candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;
            let cache = GraniteMoeHybridCache::new(!args.no_kv_cache, dtype, &config, &device)?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            (
                GraniteMoeHybrid::load(vb, &config)?,
                tokenizer_filename,
                cache,
                config,
            )
        }
    };

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let user_prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let chat_prompt = build_chat_prompt(user_prompt);
    let mut tokens = tokenizer
        .encode(chat_prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let mut tokenizer = TokenOutputStream::new(tokenizer);

    println!("Starting the inference loop:");
    println!("User: {user_prompt}\n");
    print!("Assistant: ");
    let mut logits_processor =
        create_logits_processor(args.temperature, args.top_k, args.top_p, args.seed);

    let mut start_gen = Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    let use_cache_kv = cache.use_kv_cache;

    (0..args.sample_len)
        .inspect(|index| {
            // Start the timer after the first token is generated
            if *index == 1 {
                start_gen = Instant::now();
            }
        })
        .try_for_each(|index| -> Result<()> {
            let (context_size, context_index) = if use_cache_kv && index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };
            let context = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(context, &device)?.unsqueeze(0)?;
            let logits = granite
                .forward(&input, context_index, &mut cache)?
                .squeeze(0)?;

            let logits = if args.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    args.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            index_pos += context.len();

            let next_token = logits_processor.sample(&logits)?;
            token_generated += 1;
            tokens.push(next_token);

            if next_token == config.eos_token_id.unwrap_or(EOS_TOKEN_ID) {
                return Err(E::msg("EOS token found"));
            }

            if let Some(token) = tokenizer.next_token(next_token)? {
                print!("{token}");
                std::io::stdout().flush()?;
            }
            Ok(())
        })
        .unwrap_or(());

    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }

    let duration = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated - 1) as f64 / duration.as_secs_f64(),
    );
    Ok(())
}

fn create_logits_processor(
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
    seed: u64,
) -> LogitsProcessor {
    let sampling = if temperature <= 0. {
        Sampling::ArgMax
    } else {
        match (top_k, top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    };
    LogitsProcessor::from_sampling(seed, sampling)
}
