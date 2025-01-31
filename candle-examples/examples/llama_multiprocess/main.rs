// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};
use clap::{Parser, ValueEnum};

use candle::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::LlamaEosToks;
use cudarc::driver::safe::CudaDevice;
use cudarc::nccl::safe::{Comm, Id};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::io::Write;
use std::rc::Rc;

mod model;
use model::{Config, Llama};

const MAX_SEQ_LEN: usize = 4096;
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    V2_7b,
    V2_70b,
    V3_8b,
    V3_70b,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    num_shards: usize,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

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

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    dtype: Option<String>,

    #[arg(long, default_value = "v3-8b")]
    which: Which,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();

    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => match args.which {
            Which::V2_7b | Which::V2_70b => DType::F16,
            Which::V3_8b | Which::V3_70b => DType::BF16,
        },
    };

    let comm_file = std::path::PathBuf::from(&args.comm_file);
    if comm_file.exists() {
        bail!("comm file {comm_file:?} already exists, please remove it first")
    }

    let api = Api::new()?;
    let model_id = match args.model_id {
        Some(model) => model,
        None => match args.which {
            Which::V2_7b => "meta-llama/Llama-2-7b-hf".to_string(),
            Which::V2_70b => "meta-llama/Llama-2-70b-hf".to_string(),
            Which::V3_8b => "meta-llama/Meta-Llama-3-8B".to_string(),
            Which::V3_70b => "meta-llama/Meta-Llama-3-70B".to_string(),
        },
    };
    println!("loading the model weights from {model_id}");
    let revision = args.revision.unwrap_or("main".to_string());
    let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let config_filename = api.get("config.json")?;
    let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let tokenizer_filename = api.get("tokenizer.json")?;
    let filenames = candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?;

    let num_shards = args.num_shards;
    let devices = (0..num_shards)
        .into_iter()
        .map(|rank| Device::new_cuda(rank))
        .collect::<candle::Result<Vec<_>>>()?;

    let id = Id::new().unwrap();
    let comms = devices
        .par_iter()
        .enumerate()
        .map(|(rank, device)| {
            use cudarc::driver::result;
            unsafe { result::ctx::set_current(*device.as_cuda_device()?.cu_primary_ctx()) }
                .unwrap();

            Ok(Comm::from_rank(device.as_cuda_device()?, rank, num_shards, id).unwrap())
        })
        .collect::<candle::Result<Vec<_>>>()?;

    let mut models = devices
        .par_iter()
        .zip(comms)
        .map(|(device, comm)| {
            use cudarc::driver::result;
            unsafe { result::ctx::set_current(*device.as_cuda_device()?.cu_primary_ctx()) }
                .unwrap();

            let cache = model::Cache::new(dtype, &config, &device)?;

            let vb = unsafe {
                candle_nn::var_builder::ShardedSafeTensors::var_builder(&filenames, dtype, &device)?
            };
            Llama::load(vb, &cache, &config, comm)
        })
        .collect::<candle::Result<Vec<_>>>()?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);

    println!("starting the inference loop");
    let temperature = if args.temperature <= 0. {
        None
    } else {
        Some(args.temperature)
    };
    let mut logits_processor = LogitsProcessor::new(args.seed, temperature, args.top_p);
    let mut new_tokens = vec![];
    let mut start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    for index in 0..args.sample_len {
        // Only start timing at the second token as processing the first token waits for all the
        // weights to be loaded in an async way.
        if index == 1 {
            start_gen = std::time::Instant::now()
        };
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let logits_vec = models
            .par_iter()
            .zip(&devices)
            .map(|(model, device)| {
                use cudarc::driver::result;
                unsafe { result::ctx::set_current(*device.as_cuda_device()?.cu_primary_ctx()) }
                    .unwrap();

                let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
                let logits = model.forward(&input, index_pos)?;
                logits.squeeze(0)
            })
            .collect::<candle::Result<Vec<_>>>()?;
        let logits = logits_vec[0].clone();
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        new_tokens.push(next_token);
        match config.eos_token_id {
            Some(LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                break;
            }
            Some(LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                break;
            }
            _ => (),
        }

        if rank == 0 {
            if let Some(t) = tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
    }
    println!();
    if rank == 0 {
        let dt = start_gen.elapsed();
        println!(
            "\n\n{} tokens generated ({} token/s)\n",
            args.sample_len,
            (args.sample_len - 1) as f64 / dt.as_secs_f64(),
        );
    }
    Ok(())
}
