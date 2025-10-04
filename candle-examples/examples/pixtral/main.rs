#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::pixtral::{vision_model, Config, Model};

use candle::{DType, Device, Module, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

struct TextGeneration {
    model: Model,
    image: Tensor,
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
        image: Tensor,
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
            image,
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
        let mut generated_tokens = 0usize;
        let get_token = |v| match self.tokenizer.get_token(v) {
            Some(token) => Ok(token),
            None => anyhow::bail!("cannot find the {v} token"),
        };
        let bos_token = get_token("<s>")?;
        let eos_token = get_token("</s>")?;
        let inst_token = get_token("[INST]")?;
        let end_inst_token = get_token("[/INST]")?;
        let img_break = get_token("[IMG_BREAK]")?;
        let img_end = get_token("[IMG_END]")?;
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let logits = if index > 0 {
                let context_size = if index > 0 { 1 } else { tokens.len() };
                let start_pos = tokens.len().saturating_sub(context_size);
                let ctxt = &tokens[start_pos..];
                let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
                self.model.lm_forward(&input)?
            } else {
                let (_b, _c, h, w) = self.image.dims4()?;
                let h = h / self.model.patch_size;
                let w = w / self.model.patch_size;
                let image_embeds = self.model.encode_image(&self.image)?;
                println!("generated image embeddings {image_embeds:?}");
                let image_embeds = image_embeds.to_dtype(self.model.dtype)?;
                for &t in tokens.iter() {
                    if let Some(t) = self.tokenizer.next_token(t)? {
                        print!("{t}")
                    }
                }
                std::io::stdout().flush()?;

                let break_embeds = {
                    let input = Tensor::new(&[img_break], &self.device)?.unsqueeze(0)?;
                    self.model.language_model.embed_tokens().forward(&input)?
                };
                let start_embeds = {
                    let mut in_tokens = vec![bos_token, inst_token];
                    in_tokens.extend_from_slice(tokens.as_slice());
                    let input = Tensor::new(in_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                    self.model.language_model.embed_tokens().forward(&input)?
                };
                let end_embeds = {
                    let input =
                        Tensor::new(&[img_end, end_inst_token], &self.device)?.unsqueeze(0)?;
                    self.model.language_model.embed_tokens().forward(&input)?
                };
                let mut input_embeds = vec![start_embeds];
                for h_idx in 0..h {
                    if h_idx > 0 {
                        input_embeds.push(break_embeds.clone())
                    }
                    let row = image_embeds.narrow(1, h_idx * w, w)?;
                    input_embeds.push(row);
                }
                input_embeds.push(end_embeds);

                let input_embeds = Tensor::cat(&input_embeds, 1)?;
                self.model.lm_forward_embeds(&input_embeds)?
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
            if next_token == eos_token {
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

    #[arg(long, default_value = "Describe the image.\n")]
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

    #[arg(long)]
    vision_only: bool,
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

    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.supports_bf16() && !args.vision_only {
        DType::BF16
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
    let image = image.to_device(&device)?.unsqueeze(0)?;
    println!("loaded image with shape {image:?}");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    if args.vision_only {
        let start = std::time::Instant::now();
        let model = vision_model::Model::new(&config.vision_config, vb.pp("vision_tower"))?;
        println!("loaded the model in {:?}", start.elapsed());
        let embs = model.forward(&image)?;
        println!("EMBS\n{embs}");
    } else {
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let start = std::time::Instant::now();
        let model = Model::new(&config, vb)?;
        println!("loaded the model in {:?}", start.elapsed());
        let mut pipeline = TextGeneration::new(
            model,
            image,
            tokenizer,
            args.seed,
            args.temperature,
            args.top_p,
            args.repeat_penalty,
            args.repeat_last_n,
            &device,
        );
        pipeline.run(&args.prompt, args.sample_len)?;
    }

    Ok(())
}
