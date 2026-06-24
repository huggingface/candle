#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::helium::{Config as ConfigPreview, Model as ModelPreview};
use candle_transformers::models::llama::{
    Cache as CacheV1, Llama as ModelV1, LlamaConfig as ConfigV1, LlamaEosToks,
};

use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
enum Model {
    V1 { model: ModelV1, cache: CacheV1 },
    Preview(ModelPreview),
}

impl Model {
    fn forward(&mut self, input: &Tensor, start_pos: usize) -> Result<Tensor> {
        let model = match self {
            Model::V1 { model, cache } => model.forward(input, start_pos, cache)?,
            Model::Preview(m) => m.forward(input, start_pos)?,
        };
        Ok(model)
    }
}

#[derive(Debug, Clone)]
enum Config {
    V1(ConfigV1),
    Preview(ConfigPreview),
}

impl Config {
    fn bos_token_id(&self) -> Option<u32> {
        match self {
            Config::V1(c) => c.bos_token_id,
            Config::Preview(c) => Some(c.bos_token_id),
        }
    }

    fn eos_token_id(&self) -> Option<LlamaEosToks> {
        match self {
            Config::V1(c) => c.eos_token_id.clone(),
            Config::Preview(c) => Some(LlamaEosToks::Single(c.eos_token_id)),
        }
    }
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    config: Config,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        config: Config,
        device: &Device,
    ) -> Self {
        let logits_processor = {
            let temperature = temp.unwrap_or(0.);
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (top_k, top_p) {
                    (None, None) => Sampling::GumbelSoftmax { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(seed, sampling)
        };

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
            config,
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
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
            let is_eos = self
                .config
                .eos_token_id()
                .as_ref()
                .is_some_and(|v| match v {
                    LlamaEosToks::Single(eos) => *eos == next_token,
                    LlamaEosToks::Multiple(eos) => eos.contains(&next_token),
                });
            if Some(next_token) == self.config.bos_token_id() || is_eos {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "v1-preview")]
    V1Preview,
    #[value(name = "v1")]
    V1,
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
    #[arg(long, default_value = "v1")]
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
                Which::V1Preview => "kyutai/helium-1-preview-2b",
                Which::V1 => "kyutai/helium-1-2b",
            };
            name.to_string()
        }
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames = match args.weights {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => vec![repo.get("model.safetensors")?],
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config_file = match args.config {
        Some(config_file) => std::path::PathBuf::from(config_file),
        None => repo.get("config.json")?,
    };
    let config = match args.which {
        Which::V1Preview => Config::Preview(serde_json::from_slice(&std::fs::read(config_file)?)?),
        Which::V1 => Config::V1(serde_json::from_slice(&std::fs::read(config_file)?)?),
    };
    let device = candle_examples::device(args.cpu)?;
    let (model, device) = {
        let dtype = device.bf16_default_to_f32();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let model = match &config {
            Config::V1(c) => {
                let c = c.clone().into_config(false);
                let model = ModelV1::load(vb, &c)?;
                let cache = CacheV1::new(true, dtype, &c, &device)?;
                Model::V1 { model, cache }
            }
            Config::Preview(c) => Model::Preview(ModelPreview::new(c, vb)?),
        };
        (model, device)
    };

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        Some(args.temperature),
        args.top_p,
        args.top_k,
        args.repeat_penalty,
        args.repeat_last_n,
        config,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
