// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};
use clap::Parser;

use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;

mod model;
use model::{Config, Llama, LlamaConfig};

const EOS_TOKEN: &str = "</s>";
const MAX_SEQ_LEN: usize = 4096;
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Use npy instead of safetensors
    #[arg(long)]
    npy: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long)]
    no_kv_cache: bool,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    v1: bool,

    #[arg(long)]
    use_flash_attn: bool,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online)
    #[arg(long)]
    local_weights: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;
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

    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };
    let (llama, tokenizer_filename, cache) = match args.npy {
        Some(filename) => {
            let config = if args.v1 {
                Config::config_7b_v1(args.use_flash_attn)
            } else {
                Config::config_7b_v2(args.use_flash_attn)
            };
            let cache = model::Cache::new(!args.no_kv_cache, dtype, &config, &device)?;
            let vb = VarBuilder::from_npz(filename, dtype, &device)?;
            let tokenizer = std::path::PathBuf::from("llama-tokenizer.json");
            (Llama::load(vb, &cache, &config)?, tokenizer, cache)
        }
        None => {
            let api = Api::new()?;
            let model_id = args.model_id.unwrap_or_else(|| {
                if args.v1 {
                    "Narsil/amall-7b".to_string()
                } else {
                    "meta-llama/Llama-2-7b-hf".to_string()
                }
            });
            println!("loading the model weights from {model_id}");
            let revision = args.revision.unwrap_or("main".to_string());
            let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

            let tokenizer_filename = match &args.local_weights {
                Some(path) => (path.to_owned() + "tokenizer.json").into(),
                _ => api.get("tokenizer.json")?,
            };

            let config_filename = match &args.local_weights {
                Some(path) => (path.to_owned() + "config.json").into(),
                _ => api.get("config.json")?,
            };
            let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
            let config = config.into_config(args.use_flash_attn);

            let mut filenames = vec![];
            for rfilename in [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ] {
                match &args.local_weights {
                    Some(path) => {
                        filenames.push((path.to_owned() + rfilename).into());
                    }
                    _ => {
                        let filename = api.get(rfilename)?;
                        filenames.push(filename);
                    }
                };
            }

            println!("building the model");
            let handles = filenames
                .iter()
                .map(|f| Ok(unsafe { candle::safetensors::MmapedFile::new(f.as_path())? }))
                .collect::<Result<Vec<_>>>()?;
            let tensors: Vec<_> = handles
                .iter()
                .map(|h| Ok(h.deserialize()?))
                .collect::<Result<Vec<_>>>()?;
            let cache = model::Cache::new(!args.no_kv_cache, dtype, &config, &device)?;

            let vb = VarBuilder::from_safetensors(tensors, dtype, &device);
            (Llama::load(vb, &cache, &config)?, tokenizer_filename, cache)
        }
    };
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let eos_token_id = tokenizer.token_to_id(EOS_TOKEN);
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    println!("starting the inference loop");
    print!("{prompt}");
    let mut logits_processor = LogitsProcessor::new(args.seed, args.temperature);
    let start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    for index in 0..args.sample_len {
        let context_size = if cache.use_kv_cache && index > 0 {
            1
        } else {
            tokens.len()
        };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, index_pos)?;
        let logits = logits.squeeze(0)?;
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
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        // Extracting the last token as a string is complicated, here we just apply some simple
        // heuristics as it seems to work well enough for this example. See the following for more
        // details:
        // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141
        if let Some(text) = tokenizer.id_to_token(next_token) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            print!("{text}");
            std::io::stdout().flush()?;
        }
        if Some(next_token) == eos_token_id {
            break;
        }
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        token_generated as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
