//! LFM2-VL (Liquid Foundation Model, vision-language) example.
//!
//! ```bash
//! cargo run --example lfm2-vl --release -- \
//!     --image candle-examples/examples/yolo-v8/assets/bike.jpg \
//!     --prompt "Describe this image."
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Tensor};
use candle_examples::hub::Api;
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::lfm2::Cache;
use candle_transformers::models::lfm2_vl::{Lfm2VlConfig, Lfm2VlModel};
use tokenizers::Tokenizer;

mod processor;
use processor::Lfm2VlProcessorConfig;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "lfm2-vl-450m")]
    Lfm2Vl450M,
    #[value(name = "lfm2-vl-1.6b")]
    Lfm2Vl1_6B,
    #[value(name = "lfm2-vl-3b")]
    Lfm2Vl3B,
    #[value(name = "lfm2.5-vl-1.6b")]
    Lfm2_5Vl1_6B,
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Which::Lfm2Vl450M => "LiquidAI/LFM2-VL-450M",
            Which::Lfm2Vl1_6B => "LiquidAI/LFM2-VL-1.6B",
            Which::Lfm2Vl3B => "LiquidAI/LFM2-VL-3B",
            Which::Lfm2_5Vl1_6B => "LiquidAI/LFM2.5-VL-1.6B",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Use flash attention (requires CUDA and flash-attn feature).
    #[arg(long)]
    use_flash_attn: bool,

    /// The image to describe.
    #[arg(long)]
    image: String,

    /// The question about the image.
    #[arg(long, default_value = "Describe this image.")]
    prompt: String,

    /// The temperature used to generate samples, greedy decoding when not set.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 512)]
    sample_len: usize,

    /// The model variant to use.
    #[arg(long, default_value = "lfm2-vl-450m")]
    which: Which,

    /// Override the model ID from HuggingFace Hub.
    #[arg(long)]
    model_id: Option<String>,

    /// Revision to use from HuggingFace Hub.
    #[arg(long, default_value = "main")]
    revision: String,

    /// The dtype to use: f32, f16 or bf16. Defaults to bf16 on CUDA and Metal, f32 on CPU.
    #[arg(long)]
    dtype: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.05)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    use std::io::Write;
    let args = Args::parse();

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = args
        .model_id
        .unwrap_or_else(|| args.which.model_id().to_string());
    let repo = api.model(model_id.clone()).with_revision(args.revision);
    let tokenizer = Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(E::msg)?;
    let config: Lfm2VlConfig = serde_json::from_slice(&std::fs::read(repo.get("config.json")?)?)?;
    let processor: Lfm2VlProcessorConfig = match repo.get("preprocessor_config.json") {
        Ok(file) => serde_json::from_slice(&std::fs::read(file)?)?,
        // Newer checkpoints (LFM2.5-VL) nest it in `processor_config.json`.
        Err(_) => {
            let file = repo.get("processor_config.json")?;
            let json: serde_json::Value = serde_json::from_slice(&std::fs::read(file)?)?;
            serde_json::from_value(json["image_processor"].clone())?
        }
    };
    let filenames =
        match candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json") {
            Ok(files) => files,
            Err(_) => vec![repo.get("model.safetensors")?],
        };
    println!("retrieved the files in {:?}", start.elapsed());

    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f32") => DType::F32,
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some(dtype) => anyhow::bail!("unsupported dtype {dtype}"),
        None => device.bf16_default_to_f32(),
    };

    let start = std::time::Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = Lfm2VlModel::new(&config, args.use_flash_attn, vb)?;
    let text_config = config.text_config.clone().into_config(args.use_flash_attn);
    let mut cache = Cache::new(true, dtype, &text_config, &device)?;
    println!("loaded the model in {:?}", start.elapsed());

    // Image preprocessing.
    let img = image::ImageReader::open(&args.image)?
        .decode()
        .map_err(E::msg)?
        .to_rgb8();
    let (width, height) = img.dimensions();
    let layout = processor.layout(height as usize, width as usize);
    let tiles = processor.image_tiles(&img, &layout)?;
    let (pixel_values, spatial_shapes, pixel_attention_mask) = processor.preprocess(&tiles)?;
    let pixel_values = pixel_values.to_device(&device)?.to_dtype(dtype)?;
    let pixel_attention_mask = pixel_attention_mask.to_device(&device)?;
    println!(
        "image {width}x{height}: {} tile(s) in a {}x{} grid, thumbnail: {}",
        tiles.len(),
        layout.grid.0,
        layout.grid.1,
        layout.thumbnail,
    );

    // The chat template, with `<image>` expanded into the image tokens.
    let prompt = format!(
        "<|startoftext|><|im_start|>user\n{}{}<|im_end|>\n<|im_start|>assistant\n",
        processor.image_tokens(&layout),
        args.prompt,
    );
    let mut tokens = tokenizer
        .encode(prompt, false)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let eos_token = tokenizer
        .token_to_id("<|im_end|>")
        .ok_or_else(|| E::msg("cannot find the <|im_end|> token"))?;
    let mut tokenizer = TokenOutputStream::new(tokenizer);

    let mut logits_processor = {
        let sampling = match (args.temperature, args.top_p) {
            (None, _) => Sampling::ArgMax,
            (Some(temperature), None) => Sampling::All { temperature },
            (Some(temperature), Some(p)) => Sampling::TopP { p, temperature },
        };
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    let start = std::time::Instant::now();
    let mut generated_tokens = 0usize;
    for index in 0..args.sample_len {
        // The first pass (prefill) runs the whole prompt with the image, the following ones
        // only the last generated token, the rest being in the cache.
        let (context_size, images) = if index == 0 {
            let images = (
                &pixel_values,
                spatial_shapes.as_slice(),
                &pixel_attention_mask,
            );
            (tokens.len(), Some(images))
        } else {
            (1, None)
        };
        let start_pos = tokens.len() - context_size;
        let input = Tensor::new(&tokens[start_pos..], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, images, start_pos, &mut cache)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
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
        if index == 0 {
            println!("prompt: {} tokens in {:?}", tokens.len(), start.elapsed());
        }

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated_tokens += 1;
        if next_token == eos_token {
            break;
        }
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    let dt = start.elapsed();
    println!(
        "\n{generated_tokens} tokens generated ({:.2} token/s)",
        generated_tokens as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
