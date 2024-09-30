#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::pixtral::{vision_model, Config};

use candle::{DType, Module};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

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

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long)]
    image: String,
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
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match &args.model_id {
        Some(model_id) => model_id.to_string(),
        None => "mistral-community/pixtral-12b".to_string(),
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let _tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cuda() {
        // Use F32 in all cases for now as we only run the vision encoder.
        DType::F32
    } else {
        DType::F32
    };
    let config: Config = match args.config_file {
        Some(config_file) => serde_json::from_slice(&std::fs::read(config_file)?)?,
        None => {
            let config_file = repo.get("config.json")?;
            serde_json::from_slice(&std::fs::read(config_file)?)?
        }
    };
    let image = if args.image.ends_with(".safetensors") {
        match candle::safetensors::load(&args.image, &device)?.remove("img") {
            None => anyhow::bail!("no img tensor in {}", args.image),
            Some(v) => v,
        }
    } else {
        candle_examples::imagenet::load_image_with_std_mean(
            &args.image,
            1024,
            &[0.48145466, 0.4578275, 0.40821073],
            &[0.26862954, 0.261_302_6, 0.275_777_1],
        )?
    };
    let image = image.to_device(&device)?.to_dtype(dtype)?.unsqueeze(0)?;
    println!("loaded image with shape {:?}", image);
    let start = std::time::Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = vision_model::Model::new(&config.vision_config, vb.pp("vision_tower"))?;
    println!("loaded the model in {:?}", start.elapsed());
    let embs = model.forward(&image)?;
    println!("EMBS\n{embs}");

    Ok(())
}
