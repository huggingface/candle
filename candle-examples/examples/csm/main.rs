#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::csm::{Config, Model};

use candle::DType;
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "1b")]
    Csm1b,
}

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
    use_flash_attn: bool,

    #[arg(long, default_value = "[0]Hey how are you doing?")]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.7)]
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
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    /// The model size to use.
    #[arg(long, default_value = "1b")]
    which: Which,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    config: Option<String>,

    #[arg(long)]
    weights: Option<String>,

    /// The mimi model weight file, in safetensor format.
    #[arg(long)]
    mimi_weights: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
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
        args.temperature, args.repeat_penalty, args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match args.model_id {
        Some(model_id) => model_id,
        None => {
            let name = match args.which {
                Which::Csm1b => "sesame/csm-1b",
            };
            name.to_string()
        }
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let filenames = match args.weights {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => vec![repo.get("model.safetensors")?],
    };
    let tokenizer_filename = match args.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => api
            .model("meta-llama/Llama-3.2-1B".to_string())
            .get("tokenizer.json")?,
    };
    let mimi_filename = match args.mimi_weights {
        Some(model) => std::path::PathBuf::from(model),
        None => Api::new()?
            .model("kyutai/mimi".to_string())
            .get("model.safetensors")?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config: Config = match args.config {
        Some(config_file) => serde_json::from_slice(&std::fs::read(config_file)?)?,
        None => {
            let config_file = repo.get("config.json")?;
            serde_json::from_slice(&std::fs::read(config_file)?)?
        }
    };
    let device = candle_examples::device(args.cpu)?;
    let (mut model, device) = {
        let dtype = device.bf16_default_to_f32();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let model = Model::new(&config, vb)?;
        (model, device)
    };
    let _mimi_model = {
        use candle_transformers::models::mimi;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[mimi_filename], DType::F32, &device)? };
        let config = mimi::Config::v0_1(None);
        mimi::Model::new(config, vb)?
    };

    println!("loaded the model in {:?}", start.elapsed());
    if args.prompt.ends_with(".safetensors") {
        let prompt = candle::safetensors::load(args.prompt, &device)?;
        let tokens = prompt
            .get("tokens")
            .expect("no tokens in prompt")
            .to_dtype(DType::U32)?;
        let mask = prompt.get("mask").expect("no mask in prompt").clone();
        println!("tokens:\n{tokens:?}");
        println!("mask:\n{mask:?}");
        let mut lp = candle_transformers::generation::LogitsProcessor::new(42, Some(0.8), None);
        let frame = model.generate_frame(&tokens, &mask, 0, &mut lp)?;
        println!("frame:\n{frame:?}");
    } else {
        let prompt = tokenizer.encode(args.prompt, true).map_err(E::msg)?;
        println!("{prompt:?}");
    }

    Ok(())
}
