#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::granite_docling::{config::Config, Model};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "ibm-granite/granite-docling-258M";
const IMAGE_TOKEN_ID: u32 = 100270;
const FAKE_TOKEN_AROUND_IMAGE: u32 = 100339;
const GLOBAL_IMG: u32 = 100340;
const START_OF_ROLE: u32 = 100264;
const END_OF_ROLE: u32 = 100265;
const END_OF_TEXT: u32 = 100257;

#[derive(Parser, Debug)]
#[command(about = "Granite-Docling: document image to structured markup")]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Use bf16 precision (default: f32).
    #[arg(long)]
    bf16: bool,

    /// The image file to process.
    #[arg(long)]
    image: String,

    /// Optional: path to a local model directory or HF model id.
    #[arg(long)]
    model_id: Option<String>,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy).
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// The prompt to use. Defaults to a generic document conversion prompt.
    #[arg(long)]
    prompt: Option<String>,
}

/// Load and preprocess an image following HF Idefics3ImageProcessor (no image splitting):
///   1. Resize to 512x512 square (stretch, matching HF behavior)
///   2. Rescale [0,255] -> [0,1], normalize with mean=0.5, std=0.5
/// Returns (1, 3, 512, 512).
fn load_image(path: &str, max_size: usize, device: &Device) -> Result<Tensor> {
    let img = image::ImageReader::open(path)?
        .decode()
        .map_err(E::msg)?;

    // Convert to RGB, then resize to exactly max_size x max_size (stretch to square)
    let img = img.to_rgb8();
    let (orig_w, orig_h) = (img.width(), img.height());
    let img = image::imageops::resize(
        &img,
        max_size as u32,
        max_size as u32,
        image::imageops::FilterType::Lanczos3,
    );
    println!("Image: {}x{} -> {}x{}", orig_w, orig_h, max_size, max_size);

    // Normalize: [0,255] -> [-1,1]
    let mut data = vec![0.0f32; 3 * max_size * max_size];
    for y in 0..max_size {
        for x in 0..max_size {
            let pixel = *img.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let idx = c * max_size * max_size + y * max_size + x;
                data[idx] = (pixel[c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
    }

    let tensor = Tensor::from_vec(data, (1, 3, max_size, max_size), device)?;
    Ok(tensor.to_dtype(DType::F32)?)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let dtype = if args.bf16 { DType::BF16 } else { DType::F32 };
    let model_id = args.model_id.as_deref().unwrap_or(MODEL_ID);

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(model_id.to_string());

    // Load config
    let config_path = repo.get("config.json")?;
    let config: Config = serde_json::from_reader(std::fs::File::open(&config_path)?)?;
    println!(
        "Config: vision {}x{} patch={}, text hidden={} layers={}, scale_factor={}",
        config.vision_config.image_size,
        config.vision_config.image_size,
        config.vision_config.patch_size,
        config.text_config.hidden_size,
        config.text_config.num_hidden_layers,
        config.scale_factor,
    );

    // Load tokenizer
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

    // Load model weights
    let model_path = repo.get("model.safetensors")?;
    println!("Loading model from {model_path:?}");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)? };
    let mut model = Model::new(&config, vb)?;
    println!("Model loaded.");

    // Load and preprocess image
    let max_size = config.vision_config.image_size;
    let pixel_values = load_image(&args.image, max_size, &device)?;
    let pixel_values = pixel_values.to_dtype(dtype)?;
    println!(
        "Image: {:?} -> tensor {:?}",
        args.image,
        pixel_values.shape()
    );

    // Build input_ids matching the Granite-Docling chat template:
    //   <|start_of_role|>user<|end_of_role|>
    //   <fake_token_around_image><image>*N<fake_token_around_image>
    //   Convert this page to docling.<|end_of_text|>\n
    //   <|start_of_role|>assistant<|end_of_role|>
    let prompt_text = args.prompt.as_deref().unwrap_or(
        "Convert this page to docling.",
    );
    let image_seq_len = config.image_seq_len();

    let user_text = tokenizer.encode("user", false).map_err(E::msg)?;
    let prompt_enc = tokenizer.encode(prompt_text, false).map_err(E::msg)?;
    let assistant_text = tokenizer.encode("assistant", false).map_err(E::msg)?;

    let mut input_ids: Vec<u32> = Vec::new();
    // <|start_of_role|>user<|end_of_role|>
    input_ids.push(START_OF_ROLE);
    input_ids.extend_from_slice(user_text.get_ids());
    input_ids.push(END_OF_ROLE);
    // <fake_token_around_image><global-img><image>*64<fake_token_around_image>
    input_ids.push(FAKE_TOKEN_AROUND_IMAGE);
    input_ids.push(GLOBAL_IMG);
    input_ids.extend(std::iter::repeat(IMAGE_TOKEN_ID).take(image_seq_len));
    input_ids.push(FAKE_TOKEN_AROUND_IMAGE);
    // prompt text
    input_ids.extend_from_slice(prompt_enc.get_ids());
    // <|end_of_text|>\n
    input_ids.push(END_OF_TEXT);
    // Encode the literal newline that the template adds after <|end_of_text|>
    let newline_enc = tokenizer.encode("\n", false).map_err(E::msg)?;
    input_ids.extend_from_slice(newline_enc.get_ids());
    // <|start_of_role|>assistant<|end_of_role|>
    input_ids.push(START_OF_ROLE);
    input_ids.extend_from_slice(assistant_text.get_ids());
    input_ids.push(END_OF_ROLE);

    let input_ids_tensor = Tensor::new(&input_ids[..], &device)?.unsqueeze(0)?;
    println!(
        "Input: {} total tokens ({} image placeholders)",
        input_ids.len(),
        image_seq_len,
    );

    // Initial forward pass with image
    let logits = model.setup(&pixel_values, &input_ids_tensor)?;
    let temperature = if args.temperature > 0.0 {
        Some(args.temperature)
    } else {
        None
    };
    let mut logits_processor =
        candle_transformers::generation::LogitsProcessor::new(42, temperature, None);

    // Get first generated token from last position
    let logits_last = logits.squeeze(0)?.get(logits.dim(1)? - 1)?;

    let mut token = logits_processor.sample(&logits_last)?;

    let mut generated = vec![token];
    print_token(&tokenizer, token);

    // Autoregressive generation loop
    for _ in 1..args.max_tokens {
        let input = Tensor::new(&[token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input)?;
        let logits_last = logits.squeeze(0)?.get(logits.dim(1)? - 1)?;
        token = logits_processor.sample(&logits_last)?;

        if token == config.eos_token_id || token == END_OF_TEXT {
            break;
        }
        generated.push(token);
        print_token(&tokenizer, token);
    }

    println!();
    println!("Generated {} tokens.", generated.len());
    Ok(())
}

fn print_token(tokenizer: &Tokenizer, token: u32) {
    if let Ok(text) = tokenizer.decode(&[token], false) {
        use std::io::Write;
        print!("{text}");
        let _ = std::io::stdout().flush();
    }
}
