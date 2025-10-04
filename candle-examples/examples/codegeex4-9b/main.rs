use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::codegeex4_9b::*;
use clap::Parser;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose: bool,
    dtype: DType,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: f64,
        top_p: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose: bool,
        device: &Device,
        dtype: DType,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, Some(temp), Some(top_p));
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose,
            device: device.clone(),
            dtype,
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> anyhow::Result<()> {
        use std::io::Write;
        println!("starting the inference loop");
        let tokens = self.tokenizer.encode(prompt, true).expect("tokens error");
        if tokens.is_empty() {
            panic!("Empty prompts are not supported in the chatglm model.")
        }
        if self.verbose {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => panic!("cannot find the endoftext token"),
        };
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;

        print!("{prompt}");
        std::io::stdout().flush().expect("output flush error");
        let start_gen = std::time::Instant::now();

        println!("\n start_gen");
        println!("samplelen {sample_len}");
        let mut count = 0;
        let mut result = vec![];
        for index in 0..sample_len {
            count += 1;
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(self.dtype)?;
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
            if next_token == eos_token {
                break;
            }
            let token = self
                .tokenizer
                .decode(&[next_token], true)
                .expect("Token error");
            if self.verbose {
                println!("[Count: {count}] [Raw Token: {next_token}] [Decode Token: {token}]");
            }
            result.push(token);
            std::io::stdout().flush()?;
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        println!("Result:");
        for tokens in result {
            print!("{tokens}");
        }
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
    #[arg(long, default_value_t = 0.95)]
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
    let model_id = match args.model_id {
        Some(model_id) => model_id.to_string(),
        None => "THUDM/codegeex4-all-9b".to_string(),
    };
    let revision = match args.revision {
        Some(rev) => rev.to_string(),
        None => "main".to_string(),
    };
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let tokenizer_filename = match args.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => api
            .model("THUDM/codegeex4-all-9b".to_string())
            .get("tokenizer.json")
            .map_err(anyhow::Error::msg)?,
    };
    let config_filename = match &args.weight_path {
        Some(path) => std::path::Path::new(path).join("config.json"),
        None => repo.get("config.json")?,
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

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        args.verbose,
        &device,
        dtype,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
