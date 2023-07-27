#![allow(unused)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use clap::Parser;

mod model;
use model::{Config, GPTBigCode};

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

struct TextGeneration {
    model: GPTBigCode,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
}

impl TextGeneration {
    fn new(
        model: GPTBigCode,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp);
        Self {
            model,
            tokenizer,
            logits_processor,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        println!("starting the inference loop");
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut new_tokens = vec![];
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let start_gen = std::time::Instant::now();
            let context_size = if self.model.config().use_cache && index > 0 {
                1
            } else {
                tokens.len()
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            new_tokens.push(next_token);
            println!("> {:?}", start_gen.elapsed());
            println!(
                "{} token: {} '{}'",
                index + 1,
                next_token,
                self.tokenizer
                    .decode(vec![next_token], true)
                    .map_err(E::msg)?
            );
        }
        let dt = start_gen.elapsed();
        println!(
            "{sample_len} tokens generated ({} token/s)\n----\n{}\n----",
            sample_len as f64 / dt.as_secs_f64(),
            self.tokenizer.decode(new_tokens, true).map_err(E::msg)?
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

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    #[arg(long, default_value = "bigcode/starcoder")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = Repo::with_revision(args.model_id, RepoType::Model, args.revision);
    let tokenizer_filename = api.get(&repo, "tokenizer.json")?;
    let mut filenames = vec![];
    for rfilename in [] {
        let filename = api.get(&repo, rfilename)?;
        filenames.push(filename);
    }
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let weights = filenames
        .iter()
        .map(|f| Ok(unsafe { candle::safetensors::MmapedFile::new(f)? }))
        .collect::<Result<Vec<_>>>()?;
    let weights = weights
        .iter()
        .map(|f| Ok(f.deserialize()?))
        .collect::<Result<Vec<_>>>()?;

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;
    let vb = VarBuilder::from_safetensors(weights, DType::F32, &device);
    let config = Config::starcoder();
    let model = GPTBigCode::load(vb, config)?;
    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(model, tokenizer, args.seed, args.temperature, &device);
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
