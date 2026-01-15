//! HunyuanOCR: Vision-Language Model for Document OCR.
//!
//! HunyuanOCR combines a Vision Transformer encoder with a Transformer decoder
//! for high-quality document OCR tasks.
//!
//! ```bash
//! # Basic OCR
//! cargo run --example hunyuan-ocr --release --features cuda -- \
//!     --image document.png
//!
//! # Custom prompt
//! cargo run --example hunyuan-ocr --release --features cuda -- \
//!     --image document.png \
//!     --prompt "Extract all text from this image"
//!
//! # Multi-image OCR
//! cargo run --example hunyuan-ocr --release --features cuda -- \
//!     --image page1.png --image page2.png
//!
//! # Batch mode
//! cargo run --example hunyuan-ocr --release --features cuda -- \
//!     --batch doc1.png doc2.png doc3.png
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::hunyuan_ocr::{Config, HunyuanOCRModel};
use clap::Parser;
use tokenizers::Tokenizer;

const DEFAULT_MODEL_ID: &str = "tencent/HunyuanOCR";

// CLIP normalization parameters
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to image(s). Can specify multiple times for multi-image processing.
    #[arg(long, num_args = 1..)]
    image: Vec<String>,

    /// Batch mode: process multiple images sequentially without reloading model.
    #[arg(long, num_args = 1..)]
    batch: Vec<String>,

    /// Prompt text for OCR
    #[arg(long, default_value = "Extract text from the image")]
    prompt: String,

    /// Model repository or local path
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model_id: String,

    /// Model revision
    #[arg(long, default_value = "main")]
    revision: String,

    /// Run on CPU rather than GPU
    #[arg(long)]
    cpu: bool,

    /// Maximum generation length
    #[arg(long, default_value = "1024")]
    max_length: usize,

    /// Use bfloat16 precision
    #[arg(long)]
    bf16: bool,

    /// Enable Flash Attention (requires CUDA and flash-attn feature)
    #[arg(long)]
    flash_attn: bool,
}

/// Smart resize algorithm matching PyTorch's HunyuanOCR processor.
///
/// Rescales the image so that:
/// 1. Both dimensions are divisible by `factor` (patch_size × spatial_merge_size = 32)
/// 2. Total pixels are within [min_pixels, max_pixels] range
/// 3. Aspect ratio is maintained as closely as possible
fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    // Check aspect ratio constraint
    let aspect = if height > width {
        height as f64 / width as f64
    } else {
        width as f64 / height as f64
    };
    if aspect > 200.0 {
        return Err(E::msg(format!(
            "Aspect ratio {:.1} exceeds maximum of 200",
            aspect
        )));
    }

    // Round to nearest multiple of factor
    let mut h_bar = ((height as f64 / factor as f64).round() as usize).max(1) * factor;
    let mut w_bar = ((width as f64 / factor as f64).round() as usize).max(1) * factor;

    let total_pixels = h_bar * w_bar;

    if total_pixels > max_pixels {
        // Scale down to fit within max_pixels
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = ((height as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
        w_bar = ((width as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
    } else if total_pixels < min_pixels {
        // Scale up to meet min_pixels
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
        w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
    }

    Ok((h_bar, w_bar))
}

/// Type alias for image loading result.
type ImageLoadResult = (Tensor, Vec<(i64, i64, i64)>);

/// Load and preprocess image for HunyuanOCR.
///
/// Returns (pixel_values, grid_thw) where:
/// - pixel_values: [num_patches, 3, patch_size, patch_size]
/// - grid_thw: [(t, h_patches, w_patches)]
fn load_image(path: &str, device: &Device, dtype: DType) -> Result<ImageLoadResult> {
    let img = image::ImageReader::open(path)?
        .decode()
        .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?;

    let img = img.to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);

    // HunyuanOCR uses patch_size=16, spatial_merge_size=2, factor=32
    let patch_size = 16;
    let spatial_merge = 2;
    let factor = patch_size * spatial_merge; // 32
    let min_pixels = 512 * 512;
    let max_pixels = 2048 * 2048;

    let (new_height, new_width) = smart_resize(height, width, factor, min_pixels, max_pixels)?;

    // Resize image using high-quality CatmullRom filter
    let resized = image::imageops::resize(
        &img,
        new_width as u32,
        new_height as u32,
        image::imageops::FilterType::CatmullRom,
    );

    // Convert to tensor and apply CLIP normalization
    let h_patches = new_height / patch_size;
    let w_patches = new_width / patch_size;
    let num_patches = h_patches * w_patches;

    // Create normalized tensor in CHW format
    let mut data = vec![0f32; 3 * new_height * new_width];
    for c in 0..3 {
        for y in 0..new_height {
            for x in 0..new_width {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let idx = c * new_height * new_width + y * new_width + x;
                // Normalize: (pixel/255 - mean) / std
                data[idx] = (pixel[c] as f32 / 255.0 - CLIP_MEAN[c]) / CLIP_STD[c];
            }
        }
    }

    // Create tensor [3, H, W]
    let tensor = Tensor::from_vec(data, (3, new_height, new_width), &Device::Cpu)?;

    // Split into patches: [3, H, W] -> [num_patches, 3, patch_size, patch_size]
    // Reshape to [3, h_patches, patch_size, w_patches, patch_size]
    let reshaped = tensor.reshape((3, h_patches, patch_size, w_patches, patch_size))?;
    // Permute to [h_patches, w_patches, 3, patch_size, patch_size]
    let permuted = reshaped.permute((1, 3, 0, 2, 4))?;
    // Reshape to [num_patches, 3, patch_size, patch_size]
    let patches = permuted.reshape((num_patches, 3, patch_size, patch_size))?;

    // Convert to target dtype and device
    let pixel_values = patches.to_dtype(dtype)?.to_device(device)?;

    // Grid THW: (temporal, height_patches, width_patches)
    let grid_thw = vec![(1i64, h_patches as i64, w_patches as i64)];

    println!(
        "Image: {}x{} -> {}x{} ({} x {} patches, {} total)",
        width, height, new_width, new_height, h_patches, w_patches, num_patches
    );

    Ok((pixel_values, grid_thw))
}

// Special token IDs for HunyuanOCR chat template
const BOS_TOKEN_ID: u32 = 120000; // <｜hy_begin▁of▁sentence｜>
const SYSTEM_SUFFIX_ID: u32 = 120021; // <｜hy_place▁holder▁no▁3｜>
const USER_TOKEN_ID: u32 = 120006; // <｜hy_User｜>
const EOS_TOKEN_IDS: [u32; 2] = [120007, 120020]; // Assistant marker and end_of_sentence

/// Calculate number of image tokens after Vision Encoder processing.
///
/// Formula: grid_h * (grid_w + 1) + 2
/// - grid_h and grid_w are patch counts after spatial_merge_size processing
/// - +1 is for newline token at end of each row
/// - +2 is for begin and end tokens
fn calculate_num_image_tokens(grid_thw: &[(i64, i64, i64)], spatial_merge_size: usize) -> usize {
    let mut total = 0;
    for (_t, h, w) in grid_thw {
        let grid_h = *h as usize / spatial_merge_size;
        let grid_w = *w as usize / spatial_merge_size;
        // Formula: grid_h * (grid_w + 1) + 2
        let tokens_per_image = grid_h * (grid_w + 1) + 2;
        total += tokens_per_image;
    }
    total
}

/// Build input tokens with proper HunyuanOCR chat template format.
///
/// Format: <BOS> <SYSTEM_SUFFIX> <IM_START> <IMAGE>×N <IM_END> <text> <USER_TOKEN>
fn build_input_tokens(
    tokenizer: &Tokenizer,
    prompt: &str,
    num_image_tokens: usize,
    config: &Config,
    device: &Device,
) -> Result<Tensor> {
    // Get special token IDs from config
    let im_start_id = config.im_start_id;
    let im_end_id = config.im_end_id;
    let image_token_id = config.image_token_id;

    // Tokenize prompt
    let prompt_encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;

    // Build full input sequence following HunyuanOCR chat template:
    // <BOS> <SYSTEM_SUFFIX> <IM_START> <IMAGE>×N <IM_END> <text> <USER_TOKEN>
    let mut input_ids: Vec<u32> = vec![BOS_TOKEN_ID];
    input_ids.push(SYSTEM_SUFFIX_ID);
    input_ids.push(im_start_id);
    input_ids.extend(vec![image_token_id; num_image_tokens]);
    input_ids.push(im_end_id);
    input_ids.extend(prompt_encoding.get_ids());
    input_ids.push(USER_TOKEN_ID);

    let tensor = Tensor::new(input_ids.as_slice(), device)?.unsqueeze(0)?;
    Ok(tensor)
}

/// Check if a token is an EOS token.
fn is_eos_token(token_id: u32) -> bool {
    EOS_TOKEN_IDS.contains(&token_id)
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );

    let device = candle_examples::device(args.cpu)?;
    let dtype = if args.bf16 { DType::BF16 } else { DType::F32 };
    println!("Using device: {:?}, dtype: {:?}", device, dtype);

    // Validate input
    let is_batch = !args.batch.is_empty();
    let is_image = !args.image.is_empty();

    if !is_batch && !is_image {
        return Err(E::msg("Either --image or --batch must be specified"));
    }
    if is_batch && is_image {
        return Err(E::msg(
            "Cannot combine --image and --batch. Use only one input mode.",
        ));
    }

    // Load model from HuggingFace or local path
    println!("Loading model from {}...", args.model_id);
    let start = std::time::Instant::now();

    let (config_file, tokenizer_file, model_files) = if std::path::Path::new(&args.model_id)
        .exists()
    {
        // Local path
        let model_path = std::path::PathBuf::from(&args.model_id);
        let config = model_path.join("config.json");
        let tokenizer = model_path.join("tokenizer.json");

        // Find safetensors files
        let files: Vec<_> = std::fs::read_dir(&model_path)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .map(|e| e.path())
            .collect();

        (config, tokenizer, files)
    } else {
        // HuggingFace Hub
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            args.model_id.clone(),
            hf_hub::RepoType::Model,
            args.revision.clone(),
        ));

        let config = repo.get("config.json")?;
        let tokenizer = repo.get("tokenizer.json")?;
        let files = candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;

        (config, tokenizer, files)
    };

    // Load config
    let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
    println!(
        "Vision: {}L {}H, Text: {}L {}H",
        config.vision_config.num_hidden_layers,
        config.vision_config.num_attention_heads,
        config.text_config.num_hidden_layers,
        config.text_config.num_attention_heads,
    );

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(E::msg)?;

    // Load model weights
    println!("Loading weights...");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)? };
    let mut model = HunyuanOCRModel::new(&config, args.flash_attn, vb)?;
    println!("Model loaded in {:?}", start.elapsed());

    // Get EOS token ID (check multiple EOS tokens)
    let spatial_merge = config.vision_config.spatial_merge_size;

    // Process images
    let image_paths = if is_batch { &args.batch } else { &args.image };

    for (idx, image_path) in image_paths.iter().enumerate() {
        if image_paths.len() > 1 {
            println!(
                "\n[{}/{}] Processing: {}",
                idx + 1,
                image_paths.len(),
                image_path
            );
        } else {
            println!("Processing: {}", image_path);
        }

        // Load and preprocess image
        let (pixel_values, grid_thw) = load_image(image_path, &device, dtype)?;

        // Calculate number of image tokens using correct formula
        let num_image_tokens = calculate_num_image_tokens(&grid_thw, spatial_merge);

        // Build input tokens
        let input_ids =
            build_input_tokens(&tokenizer, &args.prompt, num_image_tokens, &config, &device)?;
        let input_ids_vec: Vec<u32> = input_ids.squeeze(0)?.to_vec1()?;

        println!("Input sequence length: {}", input_ids_vec.len());

        // Clear KV cache for fresh generation
        model.clear_kv_cache();

        // Generate with xDRoPE position IDs
        println!("Generating (max {} tokens)...", args.max_length);
        let start_gen = std::time::Instant::now();

        let generated_tokens = model.generate_with_xdrope(
            &input_ids,
            &pixel_values,
            &grid_thw,
            &input_ids_vec,
            args.max_length,
            is_eos_token,
        )?;

        let gen_time = start_gen.elapsed();

        // Decode output
        let output_text = tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| E::msg(format!("Decoding error: {}", e)))?;

        // Print result
        println!("\n{:=<60}", "");
        println!("OCR Result:");
        println!("{:=<60}", "");
        println!("{}", output_text.trim());
        println!("{:=<60}", "");
        println!(
            "Generated {} tokens in {:.2}s ({:.1} tokens/sec)",
            generated_tokens.len(),
            gen_time.as_secs_f32(),
            generated_tokens.len() as f32 / gen_time.as_secs_f32()
        );
    }

    Ok(())
}
