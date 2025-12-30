//! PaddleOCR-VL: Vision-Language Model for Document Parsing.
//!
//! PaddleOCR-VL is a compact vision-language model (0.9B parameters) that combines
//! a NaViT-style visual encoder with ERNIE-4.5-0.3B for document understanding.
//!
//! Supports:
//! - Text recognition (OCR)
//! - Table recognition
//! - Formula recognition
//! - Chart recognition
//! - Multi-image processing (e.g., multi-page documents)
//! - Video processing with temporal position encoding
//!
//! ```bash
//! # Basic OCR
//! cargo run --example paddleocr-vl --release -- \
//!     --image document.png
//!
//! # Table recognition
//! cargo run --example paddleocr-vl --release -- \
//!     --image table.png \
//!     --task table
//!
//! # Formula recognition
//! cargo run --example paddleocr-vl --release -- \
//!     --image formula.png \
//!     --task formula
//!
//! # Chart recognition
//! cargo run --example paddleocr-vl --release -- \
//!     --image chart.png \
//!     --task chart
//!
//! # Multi-page document OCR (2 pages)
//! cargo run --example paddleocr-vl --release -- \
//!     --image page1.png --image page2.png
//!
//! # Video OCR (requires ffmpeg)
//! cargo run --example paddleocr-vl --release -- \
//!     --video clip.mp4 \
//!     --fps 1.0 \
//!     --max-frames 16
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::paddleocr_vl::{Config, PaddleOCRVLModel};
use clap::{Parser, ValueEnum};
use tokenizers::Tokenizer;

const DEFAULT_MODEL_ID: &str = "PaddlePaddle/PaddleOCR-VL";

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum Task {
    /// Text recognition (OCR)
    Ocr,
    /// Table recognition
    Table,
    /// Formula recognition
    Formula,
    /// Chart recognition
    Chart,
    /// Video mode - process all frames as a single video sequence (experimental)
    Video,
}

impl Task {
    fn prompt(&self) -> &'static str {
        match self {
            Task::Ocr => "OCR:",
            Task::Table => "Table Recognition:",
            Task::Formula => "Formula Recognition:",
            Task::Chart => "Chart Recognition:",
            Task::Video => "OCR:", // Video uses same prompt as OCR
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to document image(s). Can specify multiple times for multi-image processing.
    #[arg(long, num_args = 1..)]
    image: Vec<String>,

    /// Path to video file. Mutually exclusive with --image.
    #[arg(long)]
    video: Option<String>,

    /// Frames per second to extract from video (default: 1.0)
    #[arg(long, default_value = "1.0")]
    fps: f32,

    /// Maximum number of frames to extract from video (default: 16)
    #[arg(long, default_value = "16")]
    max_frames: usize,

    /// Similarity threshold for deduplication in video processing (0.0-1.0, default: 0.85)
    /// Text with similarity above this threshold to the previous frame is considered duplicate.
    #[arg(long, default_value = "0.85")]
    similarity_threshold: f32,

    /// Task type
    #[arg(long, value_enum, default_value = "ocr")]
    task: Task,

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
    #[arg(long, default_value = "1024")]
    max_length: usize,

    /// Use bfloat16 precision
    #[arg(long)]
    bf16: bool,
}

/// Compute Levenshtein distance between two strings.
///
/// Returns the minimum number of single-character edits (insertions, deletions,
/// substitutions) required to transform one string into the other.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use two rows instead of full matrix for space efficiency
    let mut prev_row: Vec<usize> = (0..=n).collect();
    let mut curr_row: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr_row[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr_row[j] = (prev_row[j] + 1) // deletion
                .min(curr_row[j - 1] + 1) // insertion
                .min(prev_row[j - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[n]
}

/// Compute normalized similarity between two strings (0.0 to 1.0).
///
/// Returns 1.0 for identical strings, 0.0 for completely different strings.
/// Uses Levenshtein distance normalized by the length of the longer string.
fn string_similarity(a: &str, b: &str) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    let distance = levenshtein_distance(a, b);
    1.0 - (distance as f32 / max_len as f32)
}

/// Result from frame-by-frame OCR processing.
#[derive(Debug, Clone)]
struct FrameOcrResult {
    /// Frame index (0-based)
    frame_index: usize,
    /// Timestamp in seconds
    timestamp: f32,
    /// Recognized text
    text: String,
}

/// Check if text is a known hallucination pattern.
///
/// Models often produce these phrases when there's no actual text to recognize
/// (e.g., empty frames, black screens, or images without text).
fn is_hallucination(text: &str) -> bool {
    let normalized = text.to_lowercase();

    // Common hallucination patterns (lowercase for comparison)
    let patterns = ["the quick brown fox jumps over the lazy dog"];

    for pattern in patterns {
        if normalized.contains(pattern) {
            return true;
        }
    }

    false
}

/// Load and preprocess image for PaddleOCR-VL.
fn load_image(path: &str, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
    let img = image::ImageReader::open(path)?
        .decode()
        .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?;

    let img = img.to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);

    // PaddleOCR-VL uses dynamic resolution with patch size 14
    // Resize to be divisible by factor (patch_size * spatial_merge = 28)
    // Use round() to match PyTorch processor's smart_resize behavior
    let patch_size = 14;
    let spatial_merge = 2;
    let factor = patch_size * spatial_merge; // 28

    // round(dim / factor) * factor
    let new_height = ((height + factor / 2) / factor) * factor;
    let new_width = ((width + factor / 2) / factor) * factor;

    // Note: PyTorch uses PIL's BICUBIC resampling which differs slightly from
    // Rust's CatmullRom. This causes minor pixel differences which may cascade
    // through transformer layers, but the model output remains correct.
    // CatmullRom is the closest match to PIL's BICUBIC among available filters.
    let resized = image::imageops::resize(
        &img,
        new_width as u32,
        new_height as u32,
        image::imageops::FilterType::CatmullRom,
    );

    // Normalize to [-1, 1] range (matching PyTorch processor output)
    // Note: PyTorch processor outputs values in [-1, 1] range despite using CLIP mean/std
    // This simpler normalization appears to match the actual output
    let mut normalized = vec![0f32; 3 * new_height * new_width];

    for c in 0..3 {
        for y in 0..new_height {
            for x in 0..new_width {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let idx = c * new_height * new_width + y * new_width + x;
                // Simple [-1, 1] normalization: 2 * (x/255) - 1
                normalized[idx] = pixel[c] as f32 / 255.0 * 2.0 - 1.0;
            }
        }
    }

    // Create tensor: (1, 3, H, W)
    let pixel_values =
        Tensor::from_vec(normalized, (1, 3, new_height, new_width), device)?.to_dtype(dtype)?;

    // Grid THW: (temporal, height_patches, width_patches)
    let h_patches = (new_height / patch_size) as u32;
    let w_patches = (new_width / patch_size) as u32;
    let grid_thw = Tensor::new(&[[1u32, h_patches, w_patches]], device)?;

    println!(
        "Image: {}x{} -> {}x{} ({} x {} patches)",
        width, height, new_width, new_height, h_patches, w_patches
    );

    Ok((pixel_values, grid_thw))
}

/// Load and preprocess multiple images for PaddleOCR-VL.
///
/// Returns separate pixel_values tensors and grid_thw tensors for each image.
/// This allows handling images of different resolutions.
fn load_images_separate(
    paths: &[String],
    device: &Device,
    dtype: DType,
) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
    let patch_size = 14;
    let spatial_merge = 2;
    let factor = patch_size * spatial_merge; // 28

    let mut all_pixels = Vec::new();
    let mut all_grids = Vec::new();

    for (i, path) in paths.iter().enumerate() {
        let img = image::ImageReader::open(path)?
            .decode()
            .map_err(|e| E::msg(format!("Failed to decode image {}: {}", path, e)))?;

        let img = img.to_rgb8();
        let (width, height) = (img.width() as usize, img.height() as usize);

        // Resize to be divisible by factor
        let new_height = ((height + factor / 2) / factor) * factor;
        let new_width = ((width + factor / 2) / factor) * factor;

        let resized = image::imageops::resize(
            &img,
            new_width as u32,
            new_height as u32,
            image::imageops::FilterType::CatmullRom,
        );

        // Normalize to [-1, 1] range
        let mut normalized = vec![0f32; 3 * new_height * new_width];
        for c in 0..3 {
            for y in 0..new_height {
                for x in 0..new_width {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    let idx = c * new_height * new_width + y * new_width + x;
                    normalized[idx] = pixel[c] as f32 / 255.0 * 2.0 - 1.0;
                }
            }
        }

        // Create tensor: (1, 3, H, W)
        let pixel_values =
            Tensor::from_vec(normalized, (1, 3, new_height, new_width), device)?.to_dtype(dtype)?;
        all_pixels.push(pixel_values);

        // Grid THW as single-row tensor
        let h_patches = (new_height / patch_size) as u32;
        let w_patches = (new_width / patch_size) as u32;
        let grid_thw = Tensor::new(&[[1u32, h_patches, w_patches]], device)?;
        all_grids.push(grid_thw);

        println!(
            "Image {}: {}x{} -> {}x{} ({} x {} patches)",
            i + 1,
            width,
            height,
            new_width,
            new_height,
            h_patches,
            w_patches
        );
    }

    Ok((all_pixels, all_grids))
}

/// Load and preprocess video frames for PaddleOCR-VL.
///
/// Extracts frames from a video file at the specified fps and preprocesses them
/// for the vision encoder. All frames are resized to the same resolution.
///
/// # Arguments
/// * `path` - Path to video file
/// * `fps` - Target frames per second to extract
/// * `max_frames` - Maximum number of frames to extract
/// * `device` - Device for tensors
/// * `dtype` - Data type for tensors
///
/// # Returns
/// Tuple of (pixel_values, video_grid_thw) where:
/// - pixel_values: (num_patches, hidden) flattened vision patches
/// - video_grid_thw: (1, 3) = [num_frames, height_patches, width_patches]
fn load_video_frames(
    path: &str,
    fps: f32,
    max_frames: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    use std::process::Command;

    // Create temporary directory for frames
    let temp_dir = std::env::temp_dir().join(format!("paddleocr_vl_frames_{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir)?;

    // Use ffmpeg to extract frames
    let output = Command::new("ffmpeg")
        .args([
            "-i",
            path,
            "-vf",
            &format!("fps={}", fps),
            "-frames:v",
            &max_frames.to_string(),
            "-y",
            &temp_dir.join("frame_%04d.png").to_string_lossy(),
        ])
        .output()
        .map_err(|e| {
            E::msg(format!(
                "Failed to run ffmpeg: {}. Make sure ffmpeg is installed.",
                e
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Clean up temp directory
        let _ = std::fs::remove_dir_all(&temp_dir);
        return Err(E::msg(format!("ffmpeg failed: {}", stderr)));
    }

    // Find all extracted frames
    let mut frame_paths: Vec<_> = std::fs::read_dir(&temp_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .map(|e| e.path())
        .collect();
    frame_paths.sort();

    if frame_paths.is_empty() {
        let _ = std::fs::remove_dir_all(&temp_dir);
        return Err(E::msg("No frames extracted from video"));
    }

    let num_frames = frame_paths.len();
    println!("Extracted {} frames from video at {} fps", num_frames, fps);

    let patch_size = 14;
    let spatial_merge = 2;
    let factor = patch_size * spatial_merge; // 28

    // Load first frame to determine dimensions
    let first_img = image::ImageReader::open(&frame_paths[0])?
        .decode()
        .map_err(|e| E::msg(format!("Failed to decode frame: {}", e)))?;
    let first_img = first_img.to_rgb8();
    let (width, height) = (first_img.width() as usize, first_img.height() as usize);

    // Resize dimensions (same for all frames)
    let new_height = ((height + factor / 2) / factor) * factor;
    let new_width = ((width + factor / 2) / factor) * factor;
    let h_patches = new_height / patch_size;
    let w_patches = new_width / patch_size;

    println!(
        "Video frames: {}x{} -> {}x{} ({} x {} patches, {} frames)",
        width, height, new_width, new_height, h_patches, w_patches, num_frames
    );

    // Process all frames
    let mut all_normalized = Vec::with_capacity(num_frames * 3 * new_height * new_width);

    for (i, frame_path) in frame_paths.iter().enumerate() {
        let img = image::ImageReader::open(frame_path)?
            .decode()
            .map_err(|e| E::msg(format!("Failed to decode frame {}: {}", i, e)))?;
        let img = img.to_rgb8();

        let resized = image::imageops::resize(
            &img,
            new_width as u32,
            new_height as u32,
            image::imageops::FilterType::CatmullRom,
        );

        // Normalize to [-1, 1] range
        for c in 0..3 {
            for y in 0..new_height {
                for x in 0..new_width {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    all_normalized.push(pixel[c] as f32 / 255.0 * 2.0 - 1.0);
                }
            }
        }
    }

    // Clean up temp directory
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Create tensor: (num_frames, 3, H, W)
    let pixel_values = Tensor::from_vec(
        all_normalized,
        (num_frames, 3, new_height, new_width),
        device,
    )?
    .to_dtype(dtype)?;

    // Video grid THW: (1, 3) = [temporal, height_patches, width_patches]
    let video_grid_thw = Tensor::new(
        &[[num_frames as u32, h_patches as u32, w_patches as u32]],
        device,
    )?;

    Ok((pixel_values, video_grid_thw))
}

/// Build input tokens for video with proper chat format.
///
/// Format: <BOS>User: <VIDEO_START><VIDEO>×N<VIDEO_END>[task]\nAssistant:
fn build_video_input_tokens(
    tokenizer: &Tokenizer,
    task: Task,
    num_video_tokens: usize,
    video_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    device: &Device,
) -> Result<Tensor> {
    // Get BOS token
    let bos_token_id = tokenizer.token_to_id("<|begin_of_sentence|>").unwrap_or(1);

    // Build prompt parts
    let user_prefix = "User: ";
    let task_text = task.prompt();
    let assistant_prefix = "\nAssistant: ";

    // Tokenize parts
    let user_encoding = tokenizer
        .encode(user_prefix, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
    let task_encoding = tokenizer
        .encode(task_text, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
    let assistant_encoding = tokenizer
        .encode(assistant_prefix, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;

    // Build full input with VIDEO tokens
    let mut input_ids: Vec<u32> = vec![bos_token_id];
    input_ids.extend(user_encoding.get_ids());
    input_ids.push(vision_start_token_id);
    input_ids.extend(vec![video_token_id; num_video_tokens]);
    input_ids.push(vision_end_token_id);
    input_ids.extend(task_encoding.get_ids());
    input_ids.extend(assistant_encoding.get_ids());

    let tensor = Tensor::new(input_ids.as_slice(), device)?.unsqueeze(0)?;
    Ok(tensor)
}

/// Build input tokens with proper chat format.
/// Format: <|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|>...<|IMAGE_END|>[task]\nAssistant:
fn build_input_tokens(
    tokenizer: &Tokenizer,
    task: Task,
    num_image_tokens: usize,
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    device: &Device,
) -> Result<Tensor> {
    // Get BOS token
    let bos_token_id = tokenizer.token_to_id("<|begin_of_sentence|>").unwrap_or(1); // Default BOS

    // Build prompt parts
    let user_prefix = "User: ";
    let task_text = task.prompt();
    let assistant_prefix = "\nAssistant: ";

    // Tokenize parts
    let user_encoding = tokenizer
        .encode(user_prefix, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
    let task_encoding = tokenizer
        .encode(task_text, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
    let assistant_encoding = tokenizer
        .encode(assistant_prefix, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;

    // Build full input:
    // <BOS> + "User: " + <IMAGE_START> + <IMAGE_PLACEHOLDER>... + <IMAGE_END> + task + "\nAssistant: "
    let mut input_ids: Vec<u32> = vec![bos_token_id];
    input_ids.extend(user_encoding.get_ids());
    input_ids.push(vision_start_token_id);
    input_ids.extend(vec![image_token_id; num_image_tokens]);
    input_ids.push(vision_end_token_id);
    input_ids.extend(task_encoding.get_ids());
    input_ids.extend(assistant_encoding.get_ids());

    let tensor = Tensor::new(input_ids.as_slice(), device)?.unsqueeze(0)?;
    Ok(tensor)
}

/// Build input tokens for multi-image processing.
///
/// Format: <BOS>User: <IMG_START><IMG>×N1<IMG_END> <IMG_START><IMG>×N2<IMG_END> ... [task]\nAssistant:
///
/// Each image gets its own <IMAGE_START>...<IMAGE_END> block with the correct
/// number of placeholder tokens for that image's resolution.
fn build_multi_image_input_tokens(
    tokenizer: &Tokenizer,
    task: Task,
    image_token_counts: &[usize], // Number of tokens for each image
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    device: &Device,
) -> Result<Tensor> {
    // Get BOS token
    let bos_token_id = tokenizer.token_to_id("<|begin_of_sentence|>").unwrap_or(1);

    // Build prompt parts
    let user_prefix = "User: ";
    let task_text = task.prompt();
    let assistant_prefix = "\nAssistant: ";

    // Tokenize parts
    let user_encoding = tokenizer
        .encode(user_prefix, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
    let task_encoding = tokenizer
        .encode(task_text, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
    let assistant_encoding = tokenizer
        .encode(assistant_prefix, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;

    // Build full input:
    // <BOS> + "User: " + [<IMAGE_START> + <IMAGE_PLACEHOLDER>... + <IMAGE_END>] × num_images + task + "\nAssistant: "
    let mut input_ids: Vec<u32> = vec![bos_token_id];
    input_ids.extend(user_encoding.get_ids());

    // Add each image block
    for (i, &num_tokens) in image_token_counts.iter().enumerate() {
        input_ids.push(vision_start_token_id);
        input_ids.extend(vec![image_token_id; num_tokens]);
        input_ids.push(vision_end_token_id);

        // Add space between images (but not after the last one)
        if i < image_token_counts.len() - 1 {
            // Tokenize a space
            if let Ok(space_enc) = tokenizer.encode(" ", false) {
                input_ids.extend(space_enc.get_ids());
            }
        }
    }

    input_ids.extend(task_encoding.get_ids());
    input_ids.extend(assistant_encoding.get_ids());

    let tensor = Tensor::new(input_ids.as_slice(), device)?.unsqueeze(0)?;
    Ok(tensor)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;
    let dtype = if args.bf16 { DType::BF16 } else { DType::F32 };
    println!("Using device: {:?}, dtype: {:?}", device, dtype);

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
    let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
    println!(
        "Vision: {}L {}H, Text: {}L {}H (GQA: {}KV)",
        config.vision_config.num_hidden_layers,
        config.vision_config.num_attention_heads,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
    );

    // Load tokenizer
    let tokenizer_file = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(E::msg)?;

    // Load model weights
    let model_file = match repo.get("model.safetensors") {
        Ok(f) => f,
        Err(_) => repo.get("pytorch_model.bin")?,
    };

    println!("Loading weights from {:?}...", model_file);
    let vb = if model_file.extension().map_or(false, |ext| ext == "bin") {
        VarBuilder::from_pth(&model_file, dtype, &device)?
    } else {
        unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, &device)? }
    };

    let mut model = PaddleOCRVLModel::new(&config, vb)?;
    println!("Model loaded successfully");

    // Validate input: either image(s) or video, but not both
    let is_video = args.video.is_some();
    if is_video && !args.image.is_empty() {
        return Err(E::msg("Cannot specify both --image and --video"));
    }
    if !is_video && args.image.is_empty() {
        return Err(E::msg("Either --image or --video must be specified"));
    }

    // Handle video input separately
    if is_video {
        let video_path = args.video.as_ref().unwrap();
        println!("Processing video: {}", video_path);

        // Use frame-by-frame processing by default (works better for most use cases)
        // Only use experimental video mode if --task video is specified
        if args.task != Task::Video {
            println!(
                "Processing frames individually (similarity threshold: {})",
                args.similarity_threshold
            );

            // Extract frames to temp directory
            use std::process::Command;
            let temp_dir =
                std::env::temp_dir().join(format!("paddleocr_vl_frames_{}", std::process::id()));
            std::fs::create_dir_all(&temp_dir)?;

            let output = Command::new("ffmpeg")
                .args([
                    "-i",
                    video_path,
                    "-vf",
                    &format!("fps={}", args.fps),
                    "-frames:v",
                    &args.max_frames.to_string(),
                    "-y",
                    &temp_dir.join("frame_%04d.png").to_string_lossy(),
                ])
                .output()
                .map_err(|e| {
                    E::msg(format!(
                        "Failed to run ffmpeg: {}. Make sure ffmpeg is installed.",
                        e
                    ))
                })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let _ = std::fs::remove_dir_all(&temp_dir);
                return Err(E::msg(format!("ffmpeg failed: {}", stderr)));
            }

            // Find all extracted frames
            let mut frame_paths: Vec<_> = std::fs::read_dir(&temp_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
                .map(|e| e.path())
                .collect();
            frame_paths.sort();

            if frame_paths.is_empty() {
                let _ = std::fs::remove_dir_all(&temp_dir);
                return Err(E::msg("No frames extracted from video"));
            }

            println!("Extracted {} frames at {} fps", frame_paths.len(), args.fps);

            // Get EOS token ID
            let eos_token_id = tokenizer
                .token_to_id("</s>")
                .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>"))
                .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
                .unwrap_or(2);

            // Process each frame individually
            let mut results: Vec<FrameOcrResult> = Vec::new();
            let mut prev_text = String::new();

            for (frame_idx, frame_path) in frame_paths.iter().enumerate() {
                let timestamp = frame_idx as f32 / args.fps;
                print!(
                    "\rProcessing frame {}/{} (t={:.1}s)...",
                    frame_idx + 1,
                    frame_paths.len(),
                    timestamp
                );
                std::io::Write::flush(&mut std::io::stdout())?;

                // Load frame as single image
                let frame_path_str = frame_path.to_string_lossy().to_string();
                let (pixel_values, grid_thw) = load_image(&frame_path_str, &device, dtype)?;

                // Build input tokens for this frame
                let grid_thw_vec: Vec<Vec<u32>> = grid_thw.to_vec2()?;
                let g = &grid_thw_vec[0];
                let spatial_merge_size = 2;
                let num_image_tokens =
                    (g[1] as usize / spatial_merge_size) * (g[2] as usize / spatial_merge_size);

                let input_ids = build_input_tokens(
                    &tokenizer,
                    args.task,
                    num_image_tokens,
                    config.image_token_id,
                    config.vision_start_token_id,
                    config.vision_end_token_id,
                    &device,
                )?;

                // Clear KV cache for fresh generation
                model.clear_kv_cache();

                // Generate text for this frame
                let generated_tokens = model.generate(
                    &input_ids,
                    &pixel_values,
                    &grid_thw,
                    args.max_length,
                    eos_token_id,
                )?;

                // Decode text
                let output_tokens: Vec<u32> = generated_tokens
                    .into_iter()
                    .take_while(|&t| t != eos_token_id)
                    .collect();

                let text = tokenizer.decode(&output_tokens, true).unwrap_or_default();
                let text = text.trim().to_string();

                // Skip empty text and hallucinations
                if text.is_empty() || is_hallucination(&text) {
                    continue;
                }

                // Check similarity with previous text
                let similarity = string_similarity(&text, &prev_text);

                if similarity < args.similarity_threshold {
                    // Text is sufficiently different - record it
                    results.push(FrameOcrResult {
                        frame_index: frame_idx,
                        timestamp,
                        text: text.clone(),
                    });
                    prev_text = text;
                }
            }

            // Clean up temp directory
            let _ = std::fs::remove_dir_all(&temp_dir);

            // Output results
            println!("\n\n{:=<60}", "");
            println!(
                "Frame-by-Frame OCR Results ({} unique text segments):",
                results.len()
            );
            println!("{:=<60}", "");

            for result in &results {
                println!(
                    "[{:.1}s] Frame {}: {}",
                    result.timestamp, result.frame_index, result.text
                );
            }

            println!("{:=<60}\n", "");

            // Also output combined text
            if !results.is_empty() {
                println!("Combined text:");
                println!("{:-<60}", "");
                for result in &results {
                    println!("{}", result.text);
                }
                println!("{:-<60}\n", "");
            }

            return Ok(());
        }

        // Experimental video mode (--task video)
        // Processes all frames as a single video sequence with temporal position encoding
        println!("Using experimental video mode (--task video)");

        // Load video frames
        let (pixel_values_video, video_grid_thw) =
            load_video_frames(video_path, args.fps, args.max_frames, &device, dtype)?;

        // Compute number of video tokens (after spatial merge)
        let grid_thw_vec: Vec<Vec<u32>> = video_grid_thw.to_vec2()?;
        let g = &grid_thw_vec[0];
        let spatial_merge_size = 2;
        let num_video_tokens = (g[0] as usize)
            * (g[1] as usize / spatial_merge_size)
            * (g[2] as usize / spatial_merge_size);

        println!(
            "Video tokens: {} ({}t x {}h x {}w after merge)",
            num_video_tokens,
            g[0],
            g[1] as usize / spatial_merge_size,
            g[2] as usize / spatial_merge_size
        );

        // Build input tokens for video
        let input_ids = build_video_input_tokens(
            &tokenizer,
            args.task,
            num_video_tokens,
            config.video_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
            &device,
        )?;

        println!("Input sequence length: {}", input_ids.dim(1)?);
        println!("Task: {:?}", args.task);
        println!("\nGenerating (max {} tokens)...", args.max_length);

        // Get EOS token ID (same as image generation path)
        let eos_token_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>"))
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(2);

        // Generate using video method
        let generated_tokens = model.generate_video(
            &input_ids,
            &pixel_values_video,
            &video_grid_thw,
            args.fps,
            args.max_length,
            eos_token_id,
        )?;

        // Debug: print generated tokens
        println!("Generated {} tokens:", generated_tokens.len());
        for (i, &tok) in generated_tokens.iter().enumerate().take(50) {
            let tok_str = tokenizer
                .decode(&[tok], true)
                .unwrap_or_else(|_| format!("<{}>", tok));
            println!("  {}: {} = '{}'", i, tok, tok_str);
        }
        if generated_tokens.len() > 50 {
            println!("  ... ({} more tokens)", generated_tokens.len() - 50);
        }

        // Filter out any trailing tokens after EOS (shouldn't happen, but safety check)
        let output_tokens: Vec<u32> = generated_tokens
            .into_iter()
            .take_while(|&t| t != eos_token_id)
            .collect();

        let output_text = tokenizer.decode(&output_tokens, true).map_err(E::msg)?;

        println!("\n{:=<60}", "");
        println!("Video Recognition Result:");
        println!("{:=<60}", "");
        println!("{}", output_text);
        println!("{:=<60}\n", "");

        return Ok(());
    }

    // Image processing path
    let is_multi_image = args.image.len() > 1;

    // Load and preprocess image(s)
    // For multi-image, we use separate tensors to handle different resolutions
    let (pixel_values, grid_thw, pixel_values_list, grid_thw_list) = if is_multi_image {
        println!("Processing {} images...", args.image.len());
        let (pv_list, gt_list) = load_images_separate(&args.image, &device, dtype)?;
        let pv = pv_list[0].clone();
        let gt = gt_list[0].clone();
        (pv, gt, Some(pv_list), Some(gt_list))
    } else {
        println!("Processing image: {}", args.image[0]);
        let (pv, gt) = load_image(&args.image[0], &device, dtype)?;
        (pv, gt, None, None)
    };

    // Calculate number of image tokens after spatial merge
    let spatial_merge = config.vision_config.spatial_merge_size;

    // Calculate token counts for each image
    let image_token_counts: Vec<usize> = if let Some(ref gt_list) = grid_thw_list {
        // Multi-image: calculate from separate grid tensors
        gt_list
            .iter()
            .map(|gt| {
                let grid_vec: Vec<Vec<u32>> = gt.to_vec2().unwrap();
                let g = &grid_vec[0];
                let h_patches = g[1] as usize;
                let w_patches = g[2] as usize;
                (h_patches / spatial_merge) * (w_patches / spatial_merge)
            })
            .collect()
    } else {
        // Single image
        let grid_vec = grid_thw.to_vec2::<u32>()?;
        grid_vec
            .iter()
            .map(|g| {
                let h_patches = g[1] as usize;
                let w_patches = g[2] as usize;
                (h_patches / spatial_merge) * (w_patches / spatial_merge)
            })
            .collect()
    };

    let total_image_tokens: usize = image_token_counts.iter().sum();

    if is_multi_image {
        println!(
            "Image tokens: {:?} = {} total (after {}x{} merge)",
            image_token_counts, total_image_tokens, spatial_merge, spatial_merge
        );
    } else {
        println!(
            "Image tokens: {} (after {}x{} merge)",
            total_image_tokens, spatial_merge, spatial_merge
        );
    }

    // Build input tokens
    let input_ids = if is_multi_image {
        build_multi_image_input_tokens(
            &tokenizer,
            args.task,
            &image_token_counts,
            config.image_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
            &device,
        )?
    } else {
        build_input_tokens(
            &tokenizer,
            args.task,
            image_token_counts[0],
            config.image_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
            &device,
        )?
    };
    println!("Input shape: {:?}", input_ids.dims());

    // Get EOS token ID
    let eos_token_id = tokenizer
        .token_to_id("</s>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .unwrap_or(2);

    // Generate output
    println!(
        "Generating {} output (max_length={})...",
        args.task.prompt(),
        args.max_length
    );
    let start = std::time::Instant::now();

    let generated_tokens = if is_multi_image {
        // Use separate image processing for variable resolutions
        let pv_list = pixel_values_list.as_ref().unwrap();
        let gt_list = grid_thw_list.as_ref().unwrap();
        model.generate_multi_image_separate(
            &input_ids,
            pv_list,
            gt_list,
            args.max_length,
            eos_token_id,
        )?
    } else {
        model.generate(
            &input_ids,
            &pixel_values,
            &grid_thw,
            args.max_length,
            eos_token_id,
        )?
    };

    let elapsed = start.elapsed();

    // Decode tokens
    let output_text = tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| E::msg(format!("Decoding error: {}", e)))?;

    println!("\n{:=<60}", "");
    println!("Task: {:?}", args.task);
    println!("{:=<60}", "");
    println!("{}", output_text.trim());
    println!("{:=<60}", "");
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tokens/sec)",
        generated_tokens.len(),
        elapsed.as_secs_f32(),
        generated_tokens.len() as f32 / elapsed.as_secs_f32()
    );

    Ok(())
}
