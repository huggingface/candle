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

// Row/col marker token IDs (non-contiguous in vocabulary)
const ROW_COL_TOKENS: [[u32; 4]; 4] = [
    [100258, 100259, 100261, 100262], // row 1: col 1-4
    [100263, 100267, 100268, 100341], // row 2: col 1-4
    [100342, 100343, 100344, 100345], // row 3: col 1-4
    [100346, 100347, 100348, 100349], // row 4: col 1-4
];

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

    /// Disable image splitting (single 512x512 view of entire page).
    #[arg(long)]
    no_split: bool,
}

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------

/// Resize an RGB image to exactly (w, h) and normalize to [-1, 1].
/// Returns tensor of shape (3, h, w).
fn resize_and_normalize(
    img: &image::RgbImage,
    w: u32,
    h: u32,
) -> Vec<f32> {
    let resized = image::imageops::resize(img, w, h, image::imageops::FilterType::Lanczos3);
    let (w, h) = (w as usize, h as usize);
    let mut data = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel = *resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                data[c * h * w + y * w + x] = (pixel[c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
    }
    data
}

/// Split an image into tiles following HF Idefics3ImageProcessor:
///   1. Resize so longest edge = max_long_edge (2048 default)
///   2. Compute grid dimensions that fit in tile_size (512) tiles
///   3. Resize to exact grid dimensions
///   4. Extract tiles + one global view resized to tile_size x tile_size
///
/// Returns (num_tiles + 1 global, grid_rows, grid_cols, Vec<tile_data>)
/// where each tile_data is (3, tile_size, tile_size) flattened.
fn split_image(
    img: &image::RgbImage,
    tile_size: usize,
    max_long_edge: usize,
) -> (usize, usize, Vec<Vec<f32>>) {
    let (orig_w, orig_h) = (img.width() as f64, img.height() as f64);
    let tile = tile_size as f64;

    // Step 1: compute target size (longest edge = max_long_edge)
    let longest = orig_w.max(orig_h);
    let scale = (max_long_edge as f64) / longest;
    let scaled_w = (orig_w * scale).round();
    let scaled_h = (orig_h * scale).round();

    // Step 2: compute grid dimensions
    let n_cols = (scaled_w / tile).ceil() as usize;
    let n_rows = (scaled_h / tile).ceil() as usize;

    // Step 3: resize to exact grid dimensions
    let grid_w = (n_cols * tile_size) as u32;
    let grid_h = (n_rows * tile_size) as u32;
    let resized = image::imageops::resize(img, grid_w, grid_h, image::imageops::FilterType::Lanczos3);

    println!(
        "Image splitting: {}x{} -> {}x{} grid ({}x{} tiles = {} + 1 global)",
        img.width(), img.height(), grid_w, grid_h, n_rows, n_cols, n_rows * n_cols,
    );

    // Step 4: extract tiles
    let ts = tile_size as u32;
    let mut tiles: Vec<Vec<f32>> = Vec::new();
    for row in 0..n_rows {
        for col in 0..n_cols {
            let x0 = (col as u32) * ts;
            let y0 = (row as u32) * ts;
            let sub = image::imageops::crop_imm(&resized, x0, y0, ts, ts).to_image();
            tiles.push(resize_and_normalize(&sub, ts, ts));
        }
    }

    // Step 5: global view (entire image resized to tile_size x tile_size)
    let global = resize_and_normalize(img, ts, ts);
    tiles.push(global);

    (n_rows, n_cols, tiles)
}

/// Load image without splitting — single 512x512 view.
fn load_single(img: &image::RgbImage, tile_size: usize) -> Vec<Vec<f32>> {
    println!(
        "No splitting: {}x{} -> {}x{}",
        img.width(), img.height(), tile_size, tile_size,
    );
    vec![resize_and_normalize(img, tile_size as u32, tile_size as u32)]
}

/// Stack tile data into a single tensor of shape (num_tiles, 3, H, W).
fn tiles_to_tensor(
    tiles: &[Vec<f32>],
    tile_size: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let n = tiles.len();
    let tile_elems = 3 * tile_size * tile_size;
    let mut all_data = Vec::with_capacity(n * tile_elems);
    for tile in tiles {
        all_data.extend_from_slice(tile);
    }
    let tensor = Tensor::from_vec(all_data, (n, 3, tile_size, tile_size), device)?;
    Ok(tensor.to_dtype(dtype)?)
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

/// Build input_ids for split images following the Granite-Docling chat template:
///   <|start_of_role|>user<|end_of_role|>
///   <fake><row_1_col_1><image>*64 <fake><row_1_col_2><image>*64 ... \n
///   <fake><row_2_col_1><image>*64 ... \n
///   ...
///   \n<fake><global-img><image>*64<fake>
///   prompt text<|end_of_text|>\n
///   <|start_of_role|>assistant<|end_of_role|>
fn build_input_ids_split(
    tokenizer: &Tokenizer,
    prompt_text: &str,
    image_seq_len: usize,
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<u32>> {
    let user_text = tokenizer.encode("user", false).map_err(E::msg)?;
    let prompt_enc = tokenizer.encode(prompt_text, false).map_err(E::msg)?;
    let assistant_text = tokenizer.encode("assistant", false).map_err(E::msg)?;
    let newline_enc = tokenizer.encode("\n", false).map_err(E::msg)?;

    let mut ids: Vec<u32> = Vec::new();

    // <|start_of_role|>user<|end_of_role|>
    ids.push(START_OF_ROLE);
    ids.extend_from_slice(user_text.get_ids());
    ids.push(END_OF_ROLE);

    // Tile rows
    for row in 0..n_rows {
        for col in 0..n_cols {
            ids.push(FAKE_TOKEN_AROUND_IMAGE);
            ids.push(ROW_COL_TOKENS[row][col]);
            ids.extend(std::iter::repeat(IMAGE_TOKEN_ID).take(image_seq_len));
        }
        // \n after each row
        ids.extend_from_slice(newline_enc.get_ids());
    }

    // \n<fake><global-img><image>*64<fake>
    ids.extend_from_slice(newline_enc.get_ids());
    ids.push(FAKE_TOKEN_AROUND_IMAGE);
    ids.push(GLOBAL_IMG);
    ids.extend(std::iter::repeat(IMAGE_TOKEN_ID).take(image_seq_len));
    ids.push(FAKE_TOKEN_AROUND_IMAGE);

    // prompt text
    ids.extend_from_slice(prompt_enc.get_ids());

    // <|end_of_text|>\n
    ids.push(END_OF_TEXT);
    ids.extend_from_slice(newline_enc.get_ids());

    // <|start_of_role|>assistant<|end_of_role|>
    ids.push(START_OF_ROLE);
    ids.extend_from_slice(assistant_text.get_ids());
    ids.push(END_OF_ROLE);

    Ok(ids)
}

/// Build input_ids for single image (no splitting).
fn build_input_ids_single(
    tokenizer: &Tokenizer,
    prompt_text: &str,
    image_seq_len: usize,
) -> Result<Vec<u32>> {
    let user_text = tokenizer.encode("user", false).map_err(E::msg)?;
    let prompt_enc = tokenizer.encode(prompt_text, false).map_err(E::msg)?;
    let assistant_text = tokenizer.encode("assistant", false).map_err(E::msg)?;
    let newline_enc = tokenizer.encode("\n", false).map_err(E::msg)?;

    let mut ids: Vec<u32> = Vec::new();
    ids.push(START_OF_ROLE);
    ids.extend_from_slice(user_text.get_ids());
    ids.push(END_OF_ROLE);
    ids.push(FAKE_TOKEN_AROUND_IMAGE);
    ids.push(GLOBAL_IMG);
    ids.extend(std::iter::repeat(IMAGE_TOKEN_ID).take(image_seq_len));
    ids.push(FAKE_TOKEN_AROUND_IMAGE);
    ids.extend_from_slice(prompt_enc.get_ids());
    ids.push(END_OF_TEXT);
    ids.extend_from_slice(newline_enc.get_ids());
    ids.push(START_OF_ROLE);
    ids.extend_from_slice(assistant_text.get_ids());
    ids.push(END_OF_ROLE);
    Ok(ids)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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
    let tile_size = config.vision_config.image_size;
    let image_seq_len = config.image_seq_len();
    println!(
        "Config: vision {}x{} patch={}, text hidden={} layers={}, scale_factor={}",
        tile_size, tile_size,
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
    let raw_img = image::ImageReader::open(&args.image)?
        .decode()
        .map_err(E::msg)?
        .to_rgb8();

    let prompt_text = args.prompt.as_deref().unwrap_or("Convert this page to docling.");

    let (pixel_values, input_ids) = if args.no_split {
        let tiles = load_single(&raw_img, tile_size);
        let pv = tiles_to_tensor(&tiles, tile_size, dtype, &device)?;
        let ids = build_input_ids_single(&tokenizer, prompt_text, image_seq_len)?;
        (pv, ids)
    } else {
        let (n_rows, n_cols, tiles) = split_image(&raw_img, tile_size, 2048);
        let pv = tiles_to_tensor(&tiles, tile_size, dtype, &device)?;
        let ids = build_input_ids_split(&tokenizer, prompt_text, image_seq_len, n_rows, n_cols)?;
        (pv, ids)
    };

    let n_image_tokens = input_ids.iter().filter(|&&id| id == IMAGE_TOKEN_ID).count();
    println!(
        "Input: {} tokens ({} image placeholders, {} tiles)",
        input_ids.len(),
        n_image_tokens,
        pixel_values.dim(0)?,
    );

    let input_ids_tensor = Tensor::new(&input_ids[..], &device)?.unsqueeze(0)?;

    // Initial forward pass with image
    let logits = model.setup(&pixel_values, &input_ids_tensor)?;
    let temperature = if args.temperature > 0.0 {
        Some(args.temperature)
    } else {
        None
    };
    let mut logits_processor =
        candle_transformers::generation::LogitsProcessor::new(42, temperature, None);

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
