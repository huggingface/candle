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
use cudarc::driver::CudaDevice;
use cudarc::nccl::safe::{Comm, Id};
use futures::future::join_all;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use std::net::{IpAddr, SocketAddr};
use std::rc::Rc;
use tokenizers::Tokenizer;

mod model;
use model::{Config, Llama};

mod nccl_id_distribution;
use nccl_id_distribution::{get_nccl_id_from_server, run_nccl_id_server};

const MAX_SEQ_LEN: usize = 4096;
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    V2_7b,
    V2_70b,
    V3_8b,
    V3_70b,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    num_nodes: usize,

    #[arg(long)]
    node_rank: usize,

    #[arg(long)]
    master_addr: IpAddr,

    #[arg(long)]
    master_port: u16,

    #[arg(long)]
    num_gpus_per_node: usize,

    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    #[arg(long)]
    no_kv_cache: bool,

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

#[tokio::main]
async fn main() -> Result<()> {
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

    let world_size = args.num_nodes * args.num_gpus_per_node;
    let global_rank = args.node_rank * args.num_gpus_per_node;
    let num_workers = args.num_nodes - 1;

    println!(
        "Node rank: {}, Total nodes: {}",
        args.node_rank, args.num_nodes
    );

    // Initialize NCCL
    let unique_id = if args.node_rank == 0 {
        println!("Initializing NCCL ID Server on master node");
        let id = Id::new().map_err(|e| anyhow::anyhow!("NCCL error: {:?}", e))?;
        let id_clone = id.clone();
        tokio::spawn(async move {
            if let Err(e) = run_nccl_id_server(args.master_port, id_clone, num_workers).await {
                eprintln!("NCCL ID Server error: {:?}", e);
            }
        });

        id
    } else {
        println!("Worker node connecting to NCCL ID Server");
        let server_addr = SocketAddr::new(args.master_addr, args.master_port);
        get_nccl_id_from_server(server_addr).await?
    };

    println!("NCCL ID initialized, starting GPU processes");

    let handles: Vec<_> = (0..args.num_gpus_per_node)
        .map(|local_rank| {
            let rank = global_rank + local_rank;
            let args_clone = args.clone();
            let unique_id_clone = unique_id.clone();
            tokio::spawn(async move {
                if let Err(e) =
                    run_gpu_process(args_clone, dtype, rank, world_size, unique_id_clone).await
                {
                    eprintln!("GPU process error for rank {}: {:?}", rank, e);
                }
            })
        })
        .collect();

    let results = join_all(handles).await;

    for result in results {
        if let Err(e) = result {
            eprintln!("Task join error: {:?}", e);
        }
    }

    Ok(())
}

async fn run_gpu_process(
    args: Args,
    dtype: DType,
    rank: usize,
    world_size: usize,
    unique_id: Id,
) -> Result<()> {
    let num_devices = CudaDevice::count()? as usize;
    println!("Available CUDA devices: {}", num_devices);

    let local_rank = rank % num_devices;
    println!("Using local rank {} for global rank {}", local_rank, rank);

    let device = CudaDevice::new(local_rank)?;
    let comm = match Comm::from_rank(device, rank, world_size, unique_id) {
        Ok(comm) => Rc::new(comm),
        Err(err) => anyhow::bail!("nccl error {:?}", err.0),
    };

    println!("Rank {rank:?} spawned");

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

    let device = Device::new_cuda(rank)?;
    let cache = model::Cache::new(dtype, &config, &device)?;

    println!("building the model");
    let vb = unsafe {
        candle_nn::var_builder::ShardedSafeTensors::var_builder(&filenames, dtype, &device)?
    };
    let llama = Llama::load(vb, &cache, &config, comm)?;
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
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, index_pos)?;
        let logits = logits.squeeze(0)?;
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        new_tokens.push(next_token);
        if Some(next_token) == config.eos_token_id {
            break;
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
