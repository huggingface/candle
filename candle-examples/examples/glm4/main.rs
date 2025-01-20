use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::glm4::*;
use clap::Parser;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;
struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    args: Args,
    dtype: DType,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(model: Model, tokenizer: Tokenizer, args: Args, device: &Device, dtype: DType) -> Self {
        let logits_processor =
            LogitsProcessor::new(args.seed, Some(args.temperature), Some(args.top_p));
        Self {
            model,
            tokenizer,
            logits_processor,
            args,
            device: device.clone(),
            dtype,
        }
    }

    fn run(&mut self) -> anyhow::Result<()> {
        use std::io::Write;
        let args = &self.args;
        println!("starting the inference loop");

        let tokens = self
            .tokenizer
            .encode(args.prompt.to_string(), true)
            .expect("tokens error");
        if tokens.is_empty() {
            panic!("Empty prompts are not supported in the chatglm model.")
        }
        if args.verbose {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        } else {
            print!("{}", &args.prompt);
            std::io::stdout().flush()?;
        }
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => panic!("cannot find the endoftext token"),
        };
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;

        std::io::stdout().flush().expect("output flush error");
        let start_gen = std::time::Instant::now();

        for index in 0..args.sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(self.dtype)?;
            let logits = if args.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    args.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            let token = self
                .tokenizer
                .decode(&[next_token], true)
                .expect("token decode error");
            if args.verbose {
                println!(
                    "[Count: {}] [Raw Token: {}] [Decode Token: {}]",
                    generated_tokens, next_token, token
                );
            } else {
                print!("{token}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
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
    #[arg(name = "cache", short)]
    cache_path: Option<String>,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Display the token for the specified prompt.
    #[arg(long)]
    prompt: String,

    /// Display the tokens for the specified prompt and outputs.
    #[arg(long)]
    verbose: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value_t = 0.8)]
    top_p: f64,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 8192)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    weight_path: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.2)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
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
    let api = match args.cache_path.as_ref() {
        None => hf_hub::api::sync::Api::new()?,
        Some(path) => {
            hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(path.to_string().into()))
                .build()
                .map_err(anyhow::Error::msg)?
        }
    };

    let model_id = match args.model_id.as_ref() {
        Some(model_id) => model_id.to_string(),
        None => "THUDM/glm-4-9b".to_string(),
    };
    let revision = match args.revision.as_ref() {
        Some(rev) => rev.to_string(),
        None => "main".to_string(),
    };
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let tokenizer_filename = match args.tokenizer.as_ref() {
        Some(file) => std::path::PathBuf::from(file),
        None => api
            .model("THUDM/codegeex4-all-9b".to_string())
            .get("tokenizer.json")
            .map_err(anyhow::Error::msg)?,
    };
    let config_filename = match &args.weight_path {
        Some(path) => std::path::Path::new(path).join("config.json"),
        _ => repo.get("config.json")?,
    };

    let filenames = match &args.weight_path {
        Some(path) => {
            candle_examples::hub_load_local_safetensors(path, "model.safetensors.index.json")?
        }
        _ => candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
    };

    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).expect("Tokenizer Error");

    let start = std::time::Instant::now();
    let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = Model::new(&config, vb)?;

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(model, tokenizer, args, &device, dtype);
    pipeline.run()?;
    Ok(())
}
