#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::Parser;

use candle_transformers::models::metavoice::transformer;

use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;

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

    #[arg(long)]
    spk_emb: Option<String>,
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
    let mut model = transformer::Model::new(&first_stage_config, first_stage_vb)?;
    let input = Tensor::new(
        &[
            2133u32, 2153, 2320, 2388, 2307, 2434, 2158, 2160, 2328, 2305, 2150, 2169, 2165, 2327,
            2311, 2456, 2150, 2419, 2452, 2428, 2377, 2146, 2135, 2160, 2355, 2150, 2094, 2098,
            2115, 2093, 2399, 2313, 2161, 2325, 2094, 2164, 2483, 2374, 2323, 2514, 2487, 2380,
            2307, 2166, 2149, 2154, 2160, 2321, 2160, 2149, 2150, 2157, 2095, 2561,
        ],
        &device,
    )?;
    let input = Tensor::stack(&[&input, &input], 0)?;
    let spk_emb_file = match &args.spk_emb {
        Some(w) => std::path::PathBuf::from(w),
        None => repo.get("spk_emb.safetensors")?,
    };
    let spk_emb = candle::safetensors::load(&spk_emb_file, &device)?;
    let spk_emb = match spk_emb.get("spk_emb") {
        None => anyhow::bail!("missing spk_emb tensor in {spk_emb_file:?}"),
        Some(spk_emb) => spk_emb.to_dtype(DType::F32)?,
    };
    let logits = model.forward(&input, &spk_emb, 0)?;
    println!("{logits}");
    Ok(())
}
