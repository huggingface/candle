#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::{Parser, ValueEnum};

use candle_transformers::models::quantized_rwkv_v5::Model as Q5;
use candle_transformers::models::quantized_rwkv_v6::Model as Q6;
use candle_transformers::models::rwkv_v5::{Config, Model as M5, State, Tokenizer};
use candle_transformers::models::rwkv_v6::Model as M6;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};

const EOS_TOKEN_ID: u32 = 261;

enum Model {
    M5(M5),
    Q5(Q5),
    M6(M6),
    Q6(Q6),
}

impl Model {
    fn forward(&self, xs: &Tensor, state: &mut State) -> candle::Result<Tensor> {
        match self {
            Self::M5(m) => m.forward(xs, state),
            Self::Q5(m) => m.forward(xs, state),
            Self::M6(m) => m.forward(xs, state),
            Self::Q6(m) => m.forward(xs, state),
        }
    }
}

struct TextGeneration {
    model: Model,
    config: Config,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        config: Config,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            config,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        let mut tokens = self.tokenizer.encode(prompt)?;
        let mut generated_tokens = 0usize;
        let mut state = State::new(1, &self.config, &self.device)?;
        let mut next_logits = None;
        for &t in tokens.iter() {
            let input = Tensor::new(&[[t]], &self.device)?;
            let logits = self.model.forward(&input, &mut state)?;
            next_logits = Some(logits);
            print!("{}", self.tokenizer.decode(&[t])?)
        }
        std::io::stdout().flush()?;

        let start_gen = std::time::Instant::now();
        for _ in 0..sample_len {
            let logits = match next_logits.as_ref() {
                Some(logits) => logits,
                None => anyhow::bail!("cannot work on an empty prompt"),
            };
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == EOS_TOKEN_ID || next_token == 0 {
                break;
            }
            print!("{}", self.tokenizer.decode(&[next_token])?);
            std::io::stdout().flush()?;

            let input = Tensor::new(&[[next_token]], &self.device)?;
            next_logits = Some(self.model.forward(&input, &mut state)?)
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Parser, ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum Which {
    Eagle7b,
    World1b5,
    World3b,
    World6_1b6,
}

impl std::fmt::Display for Which {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Self::Eagle7b => "RWKV/v5-Eagle-7B-HF",
            Self::World1b5 => "RWKV/rwkv-5-world-1b5",
            Self::World3b => "RWKV/rwkv-5-world-3b",
            Self::World6_1b6 => "paperfun/rwkv",
        }
    }

    fn revision(&self) -> &'static str {
        match self {
            Self::Eagle7b => "refs/pr/1",
            Self::World1b5 | Self::World3b => "refs/pr/2",
            Self::World6_1b6 => "main",
        }
    }
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
    #[arg(long, short = 'n', default_value_t = 5000)]
    sample_len: usize,

    #[arg(long, default_value = "world1b5")]
    which: Which,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long)]
    quantized: bool,

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
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id
            .unwrap_or_else(|| args.which.model_id().to_string()),
        RepoType::Model,
        args.revision
            .unwrap_or_else(|| args.which.revision().to_string()),
    ));
    let tokenizer = match args.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => api
            .model("lmz/candle-rwkv".to_string())
            .get("rwkv_vocab_v20230424.json")?,
    };
    let config_filename = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("config.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            if args.quantized {
                vec![match args.which {
                    Which::World1b5 => api
                        .model("lmz/candle-rwkv".to_string())
                        .get("world1b5-q4k.gguf")?,
                    Which::World3b => api
                        .model("lmz/candle-rwkv".to_string())
                        .get("world3b-q4k.gguf")?,
                    Which::Eagle7b => api
                        .model("lmz/candle-rwkv".to_string())
                        .get("eagle7b-q4k.gguf")?,
                    Which::World6_1b6 => repo.get("rwkv-6-world-1b6-q4k.gguf")?,
                }]
            } else {
                vec![match args.which {
                    Which::World1b5 | Which::World3b | Which::Eagle7b => {
                        repo.get("model.safetensors")?
                    }
                    Which::World6_1b6 => repo.get("rwkv-6-world-1b6.safetensors")?,
                }]
            }
        }
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::new(tokenizer)?;

    let start = std::time::Instant::now();
    let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let device = candle_examples::device(args.cpu)?;
    let model = if args.quantized {
        let filename = &filenames[0];
        let vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename, &device)?;
        match args.which {
            Which::World1b5 | Which::World3b | Which::Eagle7b => Model::Q5(Q5::new(&config, vb)?),
            Which::World6_1b6 => Model::Q6(Q6::new(&config, vb)?),
        }
    } else {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device)? };
        match args.which {
            Which::World1b5 | Which::World3b | Which::Eagle7b => Model::M5(M5::new(&config, vb)?),
            Which::World6_1b6 => Model::M6(M6::new(&config, vb)?),
        }
    };
    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        config,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
