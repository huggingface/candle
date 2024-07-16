use clap::Parser;
use codegeex4_candle::codegeex4::*;


use candle_core::{DType, Device, Tensor};
use candle_core as candle;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
    dtype: DType,
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
        verbose_prompt: bool,
        device: &Device,
	dtype: DType,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
            device: device.clone(),
	    dtype,
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<(),()> {
        use std::io::Write;
        println!("starting the inference loop");
        let tokens = self.tokenizer.encode(prompt, true).expect("tokens error");
        if tokens.is_empty() {
            panic!("Empty prompts are not supported in the chatglm model.")
        }
        if self.verbose_prompt {
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
	println!("samplelen {}",sample_len);
	let mut count = 0;
	let mut result = vec!();
        for index in 0..sample_len {
	    count += 1;
	    println!("sample count {}",count);
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device).unwrap().unsqueeze(0).expect("create tensor input error");
            let logits = self.model.forward(&input).unwrap();
            let logits = logits.squeeze(0).unwrap().to_dtype(self.dtype).unwrap();
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                ).unwrap()
            };

            let next_token = self.logits_processor.sample(&logits).unwrap();
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
	    println!("raw generate token {}",next_token);
            let token = self.tokenizer.decode(&[next_token], true).expect("Token error");
            println!("[token:{token}]");
	    result.push(token);
            std::io::stdout().flush().unwrap();
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
    /// Run on CPU rather than on GPU.
    #[arg(name="cache",short,long, default_value=".")]
    cache_path: String,
    
    #[arg(long)]
    cpu: bool,

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
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 5000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    weight_file: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> Result<(),()> {
    
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
        args.temperature.unwrap_or(0.95),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    println!("cache path {}",args.cache_path);
    let api = hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(args.cache_path.into())).build().unwrap();
    
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
            .get("tokenizer.json").unwrap(),
    };
    let filenames = match args.weight_file {
        Some(weight_file) => vec![std::path::PathBuf::from(weight_file)],
        None => candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json").unwrap(),
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).expect("Tokenizer Error");

    let start = std::time::Instant::now();
    let config = Config::codegeex4();
    let device = candle_examples::device(args.cpu).unwrap();
    let dtype = if device.is_cuda() {
	DType::BF16
    } else {
	DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device).unwrap() };
    let model = Model::new(&config, vb).unwrap();

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        args.verbose_prompt,
        &device,
	dtype,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
