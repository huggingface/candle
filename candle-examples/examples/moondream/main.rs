#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::Parser;

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{generation::LogitsProcessor, models::moondream};
use std::io::Write;
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    prompt: String,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    sample_len: usize,
}

/// Loads an image from disk using the image crate, this returns a tensor with shape
/// (3, 378, 378). OpenAI normalization is applied.
pub fn load_image<P: AsRef<std::path::Path>>(p: P) -> Result<Tensor> {
    let img = image::io::Reader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?
        .resize_to_fill(378, 378, image::imageops::FilterType::Triangle); // Adjusted to 378x378
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (378, 378, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let mean =
        Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.26862954f32, 0.261_302_6, 0.275_777_1], &Device::Cpu)?
        .reshape((3, 1, 1))?;
    (data.to_dtype(candle::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.prompt.is_empty() {
        return Err(E::msg("prompt cannot be empty"));
    }

    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.model("vikhyatk/moondream2".to_string());
    let model_file = repo.get("model.safetensors").await?;
    let tokenizer = repo.get("tokenizer.json").await?;

    let device = candle_examples::device(args.cpu)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let mut logits_processor = LogitsProcessor::new(1337, None, None);

    let config = moondream::Config::v2();
    let mut model = moondream::Model::new(&config, vb)?;

    let image = load_image(args.image)?.to_device(&device)?;
    println!("Loaded image {image:?}");

    let prompt = format!("\n\nQuestion: {0}\n\nAnswer:", args.prompt);
    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;

    let image_embeds = image.unsqueeze(0)?;
    let text_input = Tensor::new(tokens.get_ids(), &device)?.unsqueeze(0)?;
    println!("text_input: {text_input:?}");
    let image_embeds = image_embeds.apply(model.vision_encoder())?;
    println!("image_embeds: {image_embeds:?}");

    let mut tokens = tokens.get_ids().to_vec();
    let mut generated_tokens = 0;

    let eos_token = match tokenizer.get_vocab(true).get("<|endoftext|>") {
        Some(token) => *token,
        None => anyhow::bail!("cannot find the endoftext token"),
    };

    let start_gen = std::time::Instant::now();
    println!("Generated text:");
    for _ in 0..args.sample_len {
        println!("input: {text_input:?}");
        let logits = model
            .text_model
            .forward_with_img(&text_input, &image_embeds)?;
        println!("{logits:?}");
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        println!("{logits:?}");

        let next_token = logits_processor.sample(&logits)?;
        println!("{next_token}");
        tokens.push(next_token);
        generated_tokens += 1;
        if next_token == eos_token {
            break;
        }
        let token = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{token}");
        std::io::stdout().flush()?;
    }

    let dt = start_gen.elapsed();
    println!(
        "\n{} tokens generated ({:.2} token/s)",
        generated_tokens,
        generated_tokens as f64 / dt.as_secs_f64()
    );

    Ok(())
}
