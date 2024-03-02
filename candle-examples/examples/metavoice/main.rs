#![allow(unused)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::metavoice::transformer;

use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};

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
    prompt: String,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    #[arg(long)]
    first_stage_weights: Option<String>,
}

fn main() -> Result<()> {
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
    let device = candle_examples::device(args.cpu)?;
    let api = Api::new()?;
    let repo = api.model("lmz/candle-metavoice".to_string());
    let first_stage_weights = match &args.first_stage_weights {
        Some(w) => std::path::PathBuf::from(w),
        None => repo.get("first_stage.safetensors")?,
    };
    let first_stage_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[first_stage_weights], DType::F32, &device)?
    };
    let first_stage_config = transformer::Config::cfg1b_v0_1();
    let model = transformer::Model::new(&first_stage_config, first_stage_vb)?;
    Ok(())
}
