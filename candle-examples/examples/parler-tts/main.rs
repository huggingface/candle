#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::Parser;

use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::parler_tts::{Config, Model};
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Display the token for the specified prompt.
    #[arg(long)]
    verbose_prompt: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(long, default_value_t = 5000)]
    sample_len: usize,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Use f16 precision for all the computations rather than f32.
    #[arg(long)]
    f16: bool,

    #[arg(long)]
    model_file: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,
}

fn main() -> anyhow::Result<()> {
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

    let start = std::time::Instant::now();
    let api = hf_hub::api::sync::Api::new()?;
    let model_id = match args.model_id {
        Some(model_id) => model_id.to_string(),
        None => "parler-tts/parler-tts-large-v1".to_string(),
    };
    let revision = match args.revision {
        Some(r) => r,
        None => "main".to_string(),
    };
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id,
        hf_hub::RepoType::Model,
        revision,
    ));
    let model_files = match args.model_file {
        Some(m) => vec![m.into()],
        None => candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };
    let config = match args.config_file {
        Some(m) => m.into(),
        None => repo.get("config.json")?,
    };
    let tokenizer = match args.tokenizer_file {
        Some(m) => m.into(),
        None => repo.get("tokenizer.json")?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cuda() || args.f16 {
        DType::F16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)? };
    let config: Config = serde_json::from_reader(std::fs::File::open(config)?)?;
    let mut model = Model::new(&config, vb)?;
    println!("loaded the model in {:?}", start.elapsed());

    let tokens = tokenizer
        .encode(args.prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let tokens = Tensor::new(tokens, &device)?.unsqueeze(0)?;
    println!("{tokens}");

    let encoded = model.text_encoder.forward(&tokens)?;
    println!("{encoded}");

    Ok(())
}
