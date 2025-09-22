#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{quantized::QuantizedBackend, BackendDevice, DType, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::{moondream, quantized_moondream},
    quantized_nn::Linear,
};
use tokenizers::Tokenizer;

enum Model<QB: QuantizedBackend> {
    Moondream(moondream::Model<QB::Storage>),
    Quantized(quantized_moondream::Model<QB>),
}

struct TextGeneration<QB: QuantizedBackend> {
    model: Model<QB>,
    device: QB::Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

impl<QB: QuantizedBackend> TextGeneration<QB> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model<QB>,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
        device: &QB::Device,
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
        }
    }

    fn run(
        &mut self,
        prompt: &str,
        image_embeds: &Tensor<QB::Storage>,
        sample_len: usize,
    ) -> Result<()>
    where
        Linear<QB>: Module<QB::Storage>,
    {
        use std::io::Write;
        println!("starting the inference loop");
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the Moondream model.")
        }
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }

        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;

        // Moondream tokenizer bos_token and eos_token is "<|endoftext|>"
        // https://huggingface.co/vikhyatk/moondream2/blob/main/special_tokens_map.json
        let special_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the special token"),
        };
        let (bos_token, eos_token) = (special_token, special_token);

        let start_gen = std::time::Instant::now();
        let mut load_t = std::time::Duration::from_secs_f64(0f64);
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = if index > 0 {
                match self.model {
                    Model::Moondream(ref mut model) => model.text_model.forward(&input)?,
                    Model::Quantized(ref mut model) => model.text_model.forward(&input)?,
                }
            } else {
                let bos_token = Tensor::new(&[bos_token], &self.device)?.unsqueeze(0)?;
                let logits = match self.model {
                    Model::Moondream(ref mut model) => {
                        model
                            .text_model
                            .forward_with_img(&bos_token, &input, image_embeds)?
                    }
                    Model::Quantized(ref mut model) => {
                        model
                            .text_model
                            .forward_with_img(&bos_token, &input, image_embeds)?
                    }
                };
                load_t = start_gen.elapsed();
                println!("load_t: {load_t:?}");
                logits
            };
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
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
            if next_token == eos_token || tokens.ends_with(&[27, 10619, 29] /* <END> */) {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            print!("{token}");
            std::io::stdout().flush()?;
        }

        let dt = start_gen.elapsed() - load_t;
        println!(
            "\ngenerated in {} seconds\n{generated_tokens} tokens generated ({:.2} token/s)",
            dt.as_secs_f64(),
            (generated_tokens - 1) as f64 / dt.as_secs_f64()
        );

        Ok(())
    }
}

#[derive(Parser)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Display the token for the specified prompt.
    #[arg(long)]
    verbose_prompt: bool,

    #[arg(long)]
    prompt: String,

    #[arg(long)]
    image: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(long, default_value_t = 5000)]
    sample_len: usize,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Use f16 precision for all the computations rather than f32.
    #[arg(long)]
    f16: bool,

    #[arg(long)]
    model_file: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,
}

/// Loads an image from disk using the image crate, this returns a tensor with shape
/// (3, 378, 378).
pub fn load_image<P: AsRef<std::path::Path>, QB: QuantizedBackend>(
    p: P,
    device: &QB::Device,
) -> candle::Result<Tensor<QB::Storage>> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?
        .resize_to_fill(378, 378, image::imageops::FilterType::Triangle); // Adjusted to 378x378
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (378, 378, 3), device)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.5f32, 0.5, 0.5], device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.5f32, 0.5, 0.5], device)?.reshape((3, 1, 1))?;
    (data.to_dtype(candle::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.cpu {
        run::<candle::QCpuStorage>(args).await?;
    } else if candle::utils::cuda_is_available() {
        #[cfg(feature = "cuda")]
        run::<candle::QCudaStorage>(args).await?;
    } else if candle::utils::metal_is_available() {
        run::<candle::QMetalStorage>(args).await?;
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        run::<candle::QCpuStorage>(args).await?;
    }
    Ok(())
}

async fn run<QB: QuantizedBackend>(args: Args) -> anyhow::Result<()>
where
    Linear<QB>: Module<QB::Storage>,
{
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

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
    let api = hf_hub::api::tokio::Api::new()?;
    let (model_id, revision) = match args.model_id {
        Some(model_id) => (model_id.to_string(), None),
        None => {
            if args.quantized {
                ("santiagomed/candle-moondream".to_string(), None)
            } else {
                (
                    "vikhyatk/moondream1".to_string(),
                    Some("f6e9da68e8f1b78b8f3ee10905d56826db7a5802"),
                )
            }
        }
    };
    let revision = match (args.revision, revision) {
        (Some(r), _) => r,
        (None, Some(r)) => r.to_string(),
        (None, None) => "main".to_string(),
    };
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id,
        hf_hub::RepoType::Model,
        revision,
    ));
    let model_file = match args.model_file {
        Some(m) => m.into(),
        None => {
            if args.quantized {
                repo.get("model-q4_0.gguf").await?
            } else {
                repo.get("model.safetensors").await?
            }
        }
    };
    let tokenizer = match args.tokenizer_file {
        Some(m) => m.into(),
        None => repo.get("tokenizer.json").await?,
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let device = QB::Device::new(0)?;
    let config = moondream::Config::v2();
    let dtype = if args.quantized {
        if args.f16 {
            anyhow::bail!("Quantized model does not support f16");
        }
        DType::F32
    } else if args.f16 {
        DType::F16
    } else {
        DType::F32
    };
    let model: Model<QB> = if args.quantized {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &model_file,
            &device,
        )?;
        let model = quantized_moondream::Model::new(&config, vb)?;
        Model::Quantized(model)
    } else {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
        let model = moondream::Model::new(&config, vb)?;
        Model::Moondream(model)
    };
    println!("loaded the model in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let image = load_image::<_, QB>(args.image, &device)?.to_dtype(dtype)?;
    let image_embeds = image.unsqueeze(0)?;
    let image_embeds = match model {
        Model::Moondream(ref m) => image_embeds.apply(m.vision_encoder())?,
        Model::Quantized(ref m) => image_embeds.apply(m.vision_encoder())?,
    };
    println!(
        "loaded and encoded the image {image:?} in {:?}",
        start.elapsed()
    );

    let prompt = format!("\n\nQuestion: {0}\n\nAnswer:", args.prompt);
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
    );
    pipeline.run(&prompt, &image_embeds, args.sample_len)?;

    Ok(())
}
