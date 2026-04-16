#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::gemma::{Config as Config1, Model as Model1};
use candle_transformers::models::gemma2::{Config as Config2, Model as Model2};
use candle_transformers::models::gemma3::{Config as Config3, Model as Model3};

use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "2b")]
    Base2B,
    #[value(name = "7b")]
    Base7B,
    #[value(name = "2b-it")]
    Instruct2B,
    #[value(name = "7b-it")]
    Instruct7B,
    #[value(name = "1.1-2b-it")]
    InstructV1_1_2B,
    #[value(name = "1.1-7b-it")]
    InstructV1_1_7B,
    #[value(name = "code-2b")]
    CodeBase2B,
    #[value(name = "code-7b")]
    CodeBase7B,
    #[value(name = "code-2b-it")]
    CodeInstruct2B,
    #[value(name = "code-7b-it")]
    CodeInstruct7B,
    #[value(name = "2-2b")]
    BaseV2_2B,
    #[value(name = "2-2b-it")]
    InstructV2_2B,
    #[value(name = "2-9b")]
    BaseV2_9B,
    #[value(name = "2-9b-it")]
    InstructV2_9B,
    #[value(name = "3-1b")]
    BaseV3_1B,
    #[value(name = "3-1b-it")]
    InstructV3_1B,
}

enum Model {
    V1(Model1),
    V2(Model2),
    V3(Model3),
}

impl Model {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> candle::Result<Tensor> {
        match self {
            Self::V1(m) => m.forward(input_ids, pos),
            Self::V2(m) => m.forward(input_ids, pos),
            Self::V3(m) => m.forward(input_ids, pos),
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
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
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
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
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
        let eos_token = match self.tokenizer.get_token("<eos>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <eos> token"),
        };

        let eot_token = match self.tokenizer.get_token("<end_of_turn>") {
            Some(token) => token,
            None => {
                println!(
                    "Warning: <end_of_turn> token not found in tokenizer, using <eos> as a backup"
                );
                eos_token
            }
        };

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
            if next_token == eos_token || next_token == eot_token {
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

    /// The model to use.
    #[arg(long, default_value = "2-2b")]
    which: Which,

    #[arg(long)]
    use_flash_attn: bool,
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
        None => match args.which {
            Which::InstructV1_1_2B => "google/gemma-1.1-2b-it".to_string(),
            Which::InstructV1_1_7B => "google/gemma-1.1-7b-it".to_string(),
            Which::Base2B => "google/gemma-2b".to_string(),
            Which::Base7B => "google/gemma-7b".to_string(),
            Which::Instruct2B => "google/gemma-2b-it".to_string(),
            Which::Instruct7B => "google/gemma-7b-it".to_string(),
            Which::CodeBase2B => "google/codegemma-2b".to_string(),
            Which::CodeBase7B => "google/codegemma-7b".to_string(),
            Which::CodeInstruct2B => "google/codegemma-2b-it".to_string(),
            Which::CodeInstruct7B => "google/codegemma-7b-it".to_string(),
            Which::BaseV2_2B => "google/gemma-2-2b".to_string(),
            Which::InstructV2_2B => "google/gemma-2-2b-it".to_string(),
            Which::BaseV2_9B => "google/gemma-2-9b".to_string(),
            Which::InstructV2_9B => "google/gemma-2-9b-it".to_string(),
            Which::BaseV3_1B => "google/gemma-3-1b-pt".to_string(),
            Which::InstructV3_1B => "google/gemma-3-1b-it".to_string(),
        },
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
    let config_filename = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("config.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => match args.which {
            Which::BaseV3_1B | Which::InstructV3_1B => vec![repo.get("model.safetensors")?],
            _ => candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
        },
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = match args.which {
        Which::Base2B
        | Which::Base7B
        | Which::Instruct2B
        | Which::Instruct7B
        | Which::InstructV1_1_2B
        | Which::InstructV1_1_7B
        | Which::CodeBase2B
        | Which::CodeBase7B
        | Which::CodeInstruct2B
        | Which::CodeInstruct7B => {
            let config: Config1 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
            let model = Model1::new(args.use_flash_attn, &config, vb)?;
            Model::V1(model)
        }
        Which::BaseV2_2B | Which::InstructV2_2B | Which::BaseV2_9B | Which::InstructV2_9B => {
            let config: Config2 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
            let model = Model2::new(args.use_flash_attn, &config, vb)?;
            Model::V2(model)
        }
        Which::BaseV3_1B | Which::InstructV3_1B => {
            let config: Config3 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
            let model = Model3::new(args.use_flash_attn, &config, vb)?;
            Model::V3(model)
        }
    };

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );

    let prompt = match args.which {
        Which::Base2B
        | Which::Base7B
        | Which::Instruct2B
        | Which::Instruct7B
        | Which::InstructV1_1_2B
        | Which::InstructV1_1_7B
        | Which::CodeBase2B
        | Which::CodeBase7B
        | Which::CodeInstruct2B
        | Which::CodeInstruct7B
        | Which::BaseV2_2B
        | Which::InstructV2_2B
        | Which::BaseV2_9B
        | Which::InstructV2_9B
        | Which::BaseV3_1B => args.prompt,
        Which::InstructV3_1B => {
            format!(
                "<start_of_turn> user\n{}<end_of_turn>\n<start_of_turn> model\n",
                args.prompt
            )
        }
    };

    pipeline.run(&prompt, args.sample_len)?;
    Ok(())
}
