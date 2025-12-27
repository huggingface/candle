//! Donut (Document Understanding Transformer) example.
//!
//! Donut is an OCR-free document understanding model that combines:
//! - Swin Transformer encoder for image processing
//! - BART decoder for text generation
//!
//! Supports tasks like:
//! - Document parsing (CORD-v2 receipt parsing)
//! - Document classification (RVL-CDIP)
//! - Document VQA (DocVQA)
//!
//! ```bash
//! # Receipt parsing with CORD-v2 model
//! cargo run --example donut --release -- \
//!     --image receipt.png \
//!     --task cord-v2
//!
//! # Document classification
//! cargo run --example donut --release -- \
//!     --image document.png \
//!     --task rvlcdip
//!
//! # Document VQA
//! cargo run --example donut --release -- \
//!     --image invoice.png \
//!     --task docvqa \
//!     --question "What is the total amount?"
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::donut::{DonutConfig, DonutModel};
use clap::Parser;
use tokenizers::Tokenizer;

const DEFAULT_MODEL_ID: &str = "naver-clova-ix/donut-base-finetuned-cord-v2";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to document image
    #[arg(long)]
    image: String,

    /// Task type: cord-v2 (receipt parsing), docvqa (question answering),
    /// rvlcdip (classification), zhtrainticket (Chinese train tickets)
    #[arg(long, default_value = "cord-v2")]
    task: String,

    /// Optional question for DocVQA task
    #[arg(long)]
    question: Option<String>,

    /// Model repository or path
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model_id: String,

    /// Model revision
    #[arg(long, default_value = "main")]
    revision: String,

    /// Run on CPU rather than GPU
    #[arg(long)]
    cpu: bool,

    /// Maximum generation length
    #[arg(long, default_value = "512")]
    max_length: usize,

    /// The seed for random sampling
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
}

/// Load and preprocess image for Donut.
fn load_image(path: &str, config: &DonutConfig, device: &Device) -> Result<Tensor> {
    let img = image::ImageReader::open(path)?
        .decode()
        .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?;

    let target_height = config.image_height() as u32;
    let target_width = config.image_width() as u32;

    // Resize maintaining aspect ratio to fit within target dimensions
    // Use Triangle (bilinear) to match HuggingFace DonutProcessor's default
    let resized = img.resize(
        target_width,
        target_height,
        image::imageops::FilterType::Triangle,
    );

    // Create a black canvas and center the resized image (HuggingFace uses black padding)
    let mut canvas =
        image::RgbImage::from_pixel(target_width, target_height, image::Rgb([0, 0, 0]));
    let x_offset = (target_width - resized.width()) / 2;
    let y_offset = (target_height - resized.height()) / 2;
    image::imageops::overlay(
        &mut canvas,
        &resized.to_rgb8(),
        x_offset.into(),
        y_offset.into(),
    );

    let rgb = canvas;
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    // Donut uses [0.5, 0.5, 0.5] normalization
    let image_mean = [0.5f32, 0.5, 0.5];
    let image_std = [0.5f32, 0.5, 0.5];

    // Normalize: (H, W, C) -> (C, H, W) with normalization
    let mut normalized = vec![0f32; 3 * height * width];

    for (c, (&mean, &std)) in image_mean.iter().zip(image_std.iter()).enumerate() {
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let idx = c * height * width + y * width + x;
                normalized[idx] = (pixel[c] as f32 / 255.0 - mean) / std;
            }
        }
    }

    // Create tensor: (1, 3, H, W)
    let tensor = Tensor::from_vec(normalized, (1, 3, height, width), device)?;
    Ok(tensor)
}

/// Build task prompt based on task type.
fn build_task_prompt(task: &str, question: Option<&str>) -> Result<String> {
    let prompt = match task {
        "cord-v2" => "<s_cord-v2>".to_string(),
        "rvlcdip" => "<s_rvlcdip>".to_string(),
        "docvqa" => {
            let q = question.ok_or_else(|| E::msg("Question required for docvqa task"))?;
            format!("<s_docvqa><s_question>{}</s_question><s_answer>", q)
        }
        "zhtrainticket" => "<s_zhtrainticket>".to_string(),
        _ => return Err(E::msg(format!("Unknown task: {}", task))),
    };
    Ok(prompt)
}

/// Post-process output for structured tasks.
fn postprocess_output(text: &str, task: &str) -> String {
    // Remove special tokens and format output
    let text = text
        .replace("</s>", "")
        .replace("<pad>", "")
        .replace("<unk>", "");

    match task {
        "cord-v2" | "zhtrainticket" => {
            // Extract content between task tags
            let start_tag = format!("<s_{}>", task);
            let end_tag = format!("</s_{}>", task);

            if let Some(start_idx) = text.find(&start_tag) {
                let content = &text[start_idx + start_tag.len()..];
                if let Some(end_idx) = content.find(&end_tag) {
                    return content[..end_idx].trim().to_string();
                }
                return content.trim().to_string();
            }
            text.trim().to_string()
        }
        "rvlcdip" => {
            // Extract classification label from <s_class><label/></s_class> format
            if let Some(start) = text.find("<s_class>") {
                let content = &text[start + 9..];
                if let Some(end) = content.find("</s_class>") {
                    let class_content = &content[..end];
                    // Handle self-closing tag format: <invoice/> -> invoice
                    if class_content.starts_with('<') && class_content.ends_with("/>") {
                        return class_content[1..class_content.len() - 2].to_string();
                    }
                    return class_content.trim().to_string();
                }
            }
            text.trim().to_string()
        }
        "docvqa" => {
            // Extract answer content
            if let Some(start) = text.find("<s_answer>") {
                let content = &text[start + 10..];
                if let Some(end) = content.find("</s_answer>") {
                    return content[..end].trim().to_string();
                }
                return content.trim().to_string();
            }
            text.trim().to_string()
        }
        _ => text.trim().to_string(),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate docvqa requires question
    if args.task == "docvqa" && args.question.is_none() {
        anyhow::bail!("--question is required for docvqa task");
    }

    let device = candle_examples::device(args.cpu)?;
    println!("Using device: {:?}", device);

    // Load model from HuggingFace
    println!("Loading model from {}...", args.model_id);
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        args.model_id.clone(),
        hf_hub::RepoType::Model,
        args.revision.clone(),
    ));

    // Load config
    let config_file = repo.get("config.json")?;
    let config: DonutConfig = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
    println!(
        "Image size: {}x{}, Encoder hidden: {}, Decoder d_model: {}",
        config.image_height(),
        config.image_width(),
        config.encoder.embed_dim,
        config.decoder.d_model
    );

    // Load tokenizer
    let tokenizer_file = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(E::msg)?;

    // Load model weights
    let model_file = match repo.get("model.safetensors") {
        Ok(f) => f,
        Err(_) => repo.get("pytorch_model.bin")?,
    };
    let vb = if model_file.extension().map_or(false, |ext| ext == "bin") {
        VarBuilder::from_pth(&model_file, DType::F32, &device)?
    } else {
        unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F32, &device)? }
    };

    let mut model = DonutModel::load(&config, vb)?;
    println!("Model loaded successfully");

    // Load and preprocess image
    println!("Processing image: {}", args.image);
    let pixel_values = load_image(&args.image, &config, &device)?;
    println!("Image tensor shape: {:?}", pixel_values.dims());

    // Build task prompt
    let task_prompt = build_task_prompt(&args.task, args.question.as_deref())?;
    println!("Task prompt: {}", task_prompt);

    // Tokenize prompt
    let encoding = tokenizer
        .encode(task_prompt.clone(), false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
    let initial_tokens: Vec<u32> = encoding.get_ids().to_vec();
    println!("Initial tokens: {:?}", initial_tokens);

    // Encode image
    println!("Encoding image...");
    let start = std::time::Instant::now();
    let encoder_output = model.encode(&pixel_values)?;
    println!(
        "Encoder output shape: {:?} ({:.2}s)",
        encoder_output.dims(),
        start.elapsed().as_secs_f32()
    );

    // Generate output using greedy decoding
    println!("Generating output (max_length={})...", args.max_length);
    let gen_start = std::time::Instant::now();

    let mut token_ids = initial_tokens.clone();
    let eos_token_id = config.decoder.eos_token_id;

    let mut logits_processor = candle_transformers::generation::LogitsProcessor::new(
        args.seed, None, // temperature
        None, // top_p
    );

    for index in 0..args.max_length {
        let context_size = if index >= 1 { 1 } else { token_ids.len() };
        let start_pos = token_ids.len().saturating_sub(context_size);
        let past_kv_len = if index >= 1 { start_pos } else { 0 };

        let decoder_input: Vec<u32> = token_ids[start_pos..].to_vec();
        let decoder_tensor = Tensor::new(decoder_input.as_slice(), &device)?.unsqueeze(0)?;

        let logits = model.decode(&decoder_tensor, &encoder_output, past_kv_len)?;
        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;

        // Greedy decoding (argmax)
        let token = logits_processor.sample(&logits)?;
        token_ids.push(token);

        if token == eos_token_id {
            break;
        }
    }

    let gen_elapsed = gen_start.elapsed();
    let num_generated = token_ids.len() - initial_tokens.len();

    // Decode tokens to text (keep special tokens for proper postprocessing)
    // Some tasks like rvlcdip output classification as special tokens (e.g., <invoice/>)
    let output_text = tokenizer
        .decode(&token_ids, false)
        .map_err(|e| E::msg(format!("Decoding error: {}", e)))?;

    // Post-process based on task
    let formatted_output = postprocess_output(&output_text, &args.task);

    println!("\n{:=<60}", "");
    println!("Task: {}", args.task);
    if let Some(q) = &args.question {
        println!("Question: {}", q);
    }
    println!("{:=<60}", "");
    println!("{}", formatted_output);
    println!("{:=<60}", "");
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tokens/sec)",
        num_generated,
        gen_elapsed.as_secs_f32(),
        num_generated as f32 / gen_elapsed.as_secs_f32()
    );

    Ok(())
}
