#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use std::collections::HashMap;

use candle::{safetensors as st, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::granite4_vision::{
    config::Config, select_best_resolution, Model,
};
use candle_transformers::models::granitemoehybrid::GraniteMoeHybridCache;
use image::DynamicImage;
#[cfg(feature = "pdf2image")]
use pdf2image::{RenderOptionsBuilder, PDF};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "ibm-granite/granite-4.0-3b-vision";
const START_OF_ROLE: u32 = 100264;
const END_OF_ROLE: u32 = 100265;
const END_OF_TEXT: u32 = 100257;

#[derive(Parser, Debug)]
#[command(about = "Granite 4.0 3B Vision: document data extraction")]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Use bf16 precision (default: f32).
    #[arg(long)]
    bf16: bool,

    /// The image file to process (PNG/JPG).
    #[cfg_attr(not(feature = "pdf2image"), arg(long))]
    #[cfg_attr(feature = "pdf2image", arg(long, required_unless_present = "pdf"))]
    image: Option<String>,

    /// PDF file to process (requires pdf2image feature and poppler).
    #[cfg(feature = "pdf2image")]
    #[arg(long, required_unless_present = "image")]
    pdf: Option<String>,

    /// Page range for PDF (e.g. "7-8" or "7"). Defaults to all pages.
    #[cfg(feature = "pdf2image")]
    #[arg(long)]
    pages: Option<String>,

    /// Path to merged model directory (base + LoRA merged).
    /// Use merge_lora.py to prepare: python merge_lora.py --output merged/
    #[arg(long)]
    model_id: Option<String>,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy).
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// Task tag to use. One of: tables_json, tables_html, tables_otsl,
    /// chart2csv, chart2code, chart2summary
    #[arg(long, default_value = "tables_json")]
    task: String,

    /// Custom prompt (overrides --task).
    #[arg(long)]
    prompt: Option<String>,

    /// Disable image tiling (process as single 384x384 tile).
    #[arg(long)]
    no_split: bool,
}

// ---------------------------------------------------------------------------
// Image preprocessing (LlavaNext-style AnyRes)
// ---------------------------------------------------------------------------

/// Resize image to fit within target dims (preserve aspect ratio), then pad to exact dims.
fn resize_and_pad(
    img: &image::RgbImage,
    target_w: u32,
    target_h: u32,
) -> image::RgbImage {
    let (orig_w, orig_h) = (img.width(), img.height());
    let scale = f64::min(
        target_w as f64 / orig_w as f64,
        target_h as f64 / orig_h as f64,
    );
    let new_w = (orig_w as f64 * scale).round() as u32;
    let new_h = (orig_h as f64 * scale).round() as u32;

    let resized =
        image::imageops::resize(img, new_w, new_h, image::imageops::FilterType::Lanczos3);

    // Center-pad to target dimensions
    let mut padded = image::RgbImage::new(target_w, target_h);
    let offset_x = (target_w - new_w) / 2;
    let offset_y = (target_h - new_h) / 2;
    image::imageops::overlay(&mut padded, &resized, offset_x as i64, offset_y as i64);
    padded
}

/// Normalize an RGB image to tensor data: (3, H, W) with mean=0.5, std=0.5 → [-1, 1].
fn normalize_to_chw(img: &image::RgbImage) -> Vec<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel = *img.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                data[c * h * w + y * w + x] = (pixel[c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
    }
    data
}

struct PreprocessedImage {
    /// (num_tiles, 3, tile_size, tile_size) - global thumbnail is tile[0] for multi-tile
    tiles: Vec<Vec<f32>>,
    /// Number of image tokens needed for this image
    num_image_tokens: usize,
    /// Original image dimensions (height, width)
    image_size: (usize, usize),
}

fn preprocess_image(
    img: &image::RgbImage,
    cfg: &Config,
    no_split: bool,
) -> Result<PreprocessedImage> {
    let tile_size = cfg.vision_config.image_size;
    let ts = tile_size as u32;
    let image_size = (img.height() as usize, img.width() as usize);

    if no_split {
        // Single tile: resize to tile_size x tile_size
        let resized =
            image::imageops::resize(img, ts, ts, image::imageops::FilterType::Lanczos3);
        let tile_data = normalize_to_chw(&resized);
        let ds_per_side = cfg.downsampled_patches_per_side();
        // Single tile: features + newline
        let num_image_tokens = ds_per_side * ds_per_side + 1;
        return Ok(PreprocessedImage {
            tiles: vec![tile_data],
            num_image_tokens,
            image_size,
        });
    }

    // AnyRes: find best resolution and tile
    let (best_h, best_w) =
        select_best_resolution(image_size, &cfg.image_grid_pinpoints);
    let grid_h = best_h / tile_size;
    let grid_w = best_w / tile_size;

    println!(
        "Image: {}x{} → best resolution {}x{} ({grid_h}x{grid_w} tiles + global)",
        img.width(),
        img.height(),
        best_w,
        best_h,
    );

    // Resize and pad to best resolution
    let padded = resize_and_pad(img, best_w as u32, best_h as u32);

    // Global thumbnail (first tile)
    let global = image::imageops::resize(img, ts, ts, image::imageops::FilterType::Lanczos3);
    let mut tiles = vec![normalize_to_chw(&global)];

    // Extract tiles (row-major order)
    for row in 0..grid_h {
        for col in 0..grid_w {
            let x0 = (col as u32) * ts;
            let y0 = (row as u32) * ts;
            let sub = image::imageops::crop_imm(&padded, x0, y0, ts, ts).to_image();
            tiles.push(normalize_to_chw(&sub));
        }
    }

    // Calculate number of image tokens
    let ds = cfg.downsampled_patches_per_side();
    let base_features = ds * ds;

    let (unpadded_features, newline_features) = compute_unpadded_features(
        image_size.0,
        image_size.1,
        ds,
        ds,
        grid_h,
        grid_w,
    );
    let num_image_tokens = unpadded_features + newline_features + base_features;

    Ok(PreprocessedImage {
        tiles,
        num_image_tokens,
        image_size,
    })
}

/// Compute the number of unpadded features and newline features for a multi-tile image.
fn compute_unpadded_features(
    orig_h: usize,
    orig_w: usize,
    patches_h: usize,
    patches_w: usize,
    grid_h: usize,
    grid_w: usize,
) -> (usize, usize) {
    let current_h = patches_h * grid_h;
    let current_w = patches_w * grid_w;
    let orig_aspect = orig_w as f64 / orig_h as f64;
    let curr_aspect = current_w as f64 / current_h as f64;

    let (final_h, final_w) = if orig_aspect > curr_aspect {
        let scale = current_w as f64 / orig_w as f64;
        let new_h = (orig_h as f64 * scale) as usize;
        let padding = (current_h - new_h) / 2;
        (current_h - 2 * padding, current_w)
    } else {
        let scale = current_h as f64 / orig_h as f64;
        let new_w = (orig_w as f64 * scale) as usize;
        let padding = (current_w - new_w) / 2;
        (current_h, current_w - 2 * padding)
    };

    let unpadded = final_h * final_w;
    let newlines = final_h;
    (unpadded, newlines)
}

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
// Prompt construction (matches HuggingFace chat_template)
// ---------------------------------------------------------------------------

const SYSTEM_MESSAGE: &str =
    "You are a helpful assistant. Please ensure responses are professional, accurate, and safe.";

/// Expand task tags into their full prompt text (matching the HF chat template).
fn expand_task_tag(task: &str) -> &'static str {
    match task {
        "tables_json" => "Identify and extract the tabls schema\n Extruct the schema of all the tables in the image sorted according to the reading order.\nThe output must be a valid JSON object containing a list of dictionaries with the following structure:\n\n                {\n                    \"dimensions\": {\n                        \"rows\": <number of data rows (excluding header rows)>,\n                        \"columns\": <number of columns>,\n                        \"header_rows\": <number of header rows>,\n                        \"total_rows\": <total number of rows including headers>\n                    },\n                    \"cells\": [\n                        {\n                        \"row\": <row index starting at 1>,\n                        \"col\": <column index starting at 1>,\n                        \"colspan\": <number of columns spanned>,\n                        \"rowspan\": <number of rows spanned>,\n                        \"type\": \"<'header' or 'data'>\",\n                        \"header_level\": <header nesting level if type=header, else omit or null>,\n                        \"content\": \"<string content of the cell>\"\n                        },\n                        ...\n                    ]\n                }",
        "tables_html" => "Identify and extract the tabls schema\n Extruct the schema of all the tables in the image sorted according to the reading order.\nThe output must be a list of valid HTML tables",
        "tables_otsl" => "Identify and extract the tabls schema\n Extruct the schema of all the tables in the image sorted according to the reading order.\nThe output must be a list of valid OTSL objects, each consists of the following fields: \n                        <fcel> - a cell with content in it\n                        <ecel> - an empty cell\n                        <lcel> - a cell that is merged with the cell to its left\n                        <ucel> - a cell that is merged with the cell above it\n                        <xcel> - a cell that is merged with both the cell above it and the cell to its left\n                        <nl> - a new line\n                        <ched> - a clumn header\n                        <otsl> - the beginning of the OTSL table\n                        </otsl> - the end of the OTSL table\n\n                        An example for an output:\n                        [\n                        <otsl><ched>first table header1<ched>first table header2<nl><fcel>data1<fcel>data2<nl><fcel>data with horizontal span<lcel><nl><fcell>data with vertical span<ecel><nl><ucel><fcel>data3<nl></otsl>,\n                        <otsl><ched>second table header1<ched>second table header2<nl><fcel>data1<fcel>data2<nl><fcel>data with horizontal span<lcel><nl><fcell>data with vertical span<ecel><nl><ucel><fcel>data3<nl></otsl>\n                        ]",
        "chart2code" => "Generate code that recreates the chart as best as possible.",
        "chart2csv" => "Please examine this chart image. Consider you are a data visualization expert, and extract the data into a CSV table.\n\nYour CSV should:\n- Include a header row with clear column names\n- Represent all data series/categories shown in the chart\n- Use numeric values that match the chart as closely as possible\n\nOutput only the CSV data, nothing else.",
        "chart2summary" => "Can you describe this chart image?",
        _ => "",
    }
}

fn build_input_ids(
    tokenizer: &Tokenizer,
    prompt_text: &str,
    num_image_tokens: usize,
    image_token_id: u32,
) -> Result<Vec<u32>> {
    let system_text = tokenizer.encode("system", false).map_err(E::msg)?;
    let system_msg = tokenizer.encode(SYSTEM_MESSAGE, false).map_err(E::msg)?;
    let user_text = tokenizer.encode("user", false).map_err(E::msg)?;
    let prompt_enc = tokenizer.encode(prompt_text, false).map_err(E::msg)?;
    let assistant_text = tokenizer.encode("assistant", false).map_err(E::msg)?;

    let mut ids: Vec<u32> = Vec::new();

    // <|start_of_role|>system<|end_of_role|>{system_message}<|end_of_text|>\n
    ids.push(START_OF_ROLE);
    ids.extend_from_slice(system_text.get_ids());
    ids.push(END_OF_ROLE);
    ids.extend_from_slice(system_msg.get_ids());
    ids.push(END_OF_TEXT);
    // \n is token 198 for this tokenizer
    let newline = tokenizer.encode("\n", false).map_err(E::msg)?;
    ids.extend_from_slice(newline.get_ids());

    // <|start_of_role|>user<|end_of_role|>
    ids.push(START_OF_ROLE);
    ids.extend_from_slice(user_text.get_ids());
    ids.push(END_OF_ROLE);

    // <image>\n tokens
    ids.extend(std::iter::repeat_n(image_token_id, num_image_tokens));

    // prompt text
    ids.extend_from_slice(prompt_enc.get_ids());

    // <|end_of_text|>\n
    ids.push(END_OF_TEXT);
    ids.extend_from_slice(newline.get_ids());

    // <|start_of_role|>assistant<|end_of_role|>
    ids.push(START_OF_ROLE);
    ids.extend_from_slice(assistant_text.get_ids());
    ids.push(END_OF_ROLE);

    Ok(ids)
}

// ---------------------------------------------------------------------------
// LoRA merging
// ---------------------------------------------------------------------------

/// Load base model weights via mmap, merge LoRA adapter deltas, return a VarBuilder.
///
/// LoRA merge: merged_weight = base_weight + lora_B @ lora_A * (alpha / rank).
/// For this model, alpha = rank = 256, so scale = 1.0.
fn merge_lora_and_load(
    base_paths: &[std::path::PathBuf],
    adapter_path: &std::path::Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    // Load base tensors via mmap (minimal memory footprint)
    let base = unsafe { candle::safetensors::MmapedSafetensors::multi(base_paths)? };

    // Load adapter tensors into memory (~900MB)
    let adapter = st::load(adapter_path, device)?;

    // Read LoRA config: alpha/rank. For this model both are 256, scale = 1.0.
    // We hardcode for now; a full implementation would parse adapter_config.json.
    let scale = 1.0f64;

    // Build mapping: base_weight_name → (lora_A_name, lora_B_name)
    let mut lora_map: HashMap<String, (String, String)> = HashMap::new();
    for name in adapter.keys() {
        if name.ends_with(".lora_A.weight") {
            let base_key = name
                .strip_prefix("base_model.model.")
                .unwrap_or(name)
                .replace(".lora_A.weight", ".weight");
            let lora_b_name = name.replace(".lora_A.weight", ".lora_B.weight");
            lora_map.insert(base_key, (name.clone(), lora_b_name));
        }
    }

    println!(
        "  Merging {} LoRA adapter pairs (scale={scale})...",
        lora_map.len()
    );

    // Load each base tensor, merge if LoRA exists, store in HashMap
    let tensor_names: Vec<String> = base.tensors().into_iter().map(|(n, _)| n).collect();
    let mut merged: HashMap<String, Tensor> = HashMap::with_capacity(tensor_names.len());
    let mut merge_count = 0;

    for name in &tensor_names {
        let tensor = base.load(name, device)?;
        let tensor = if let Some((lora_a_name, lora_b_name)) = lora_map.get(name) {
            if let (Some(lora_a), Some(lora_b)) =
                (adapter.get(lora_a_name), adapter.get(lora_b_name))
            {
                // Compute delta = lora_B @ lora_A in F32 for precision
                let delta = lora_b
                    .to_dtype(DType::F32)?
                    .matmul(&lora_a.to_dtype(DType::F32)?)?;
                let delta = if (scale - 1.0).abs() > f64::EPSILON {
                    (delta * scale)?
                } else {
                    delta
                };
                let result = (tensor.to_dtype(DType::F32)? + delta)?;
                merge_count += 1;
                result.to_dtype(dtype)?
            } else {
                tensor.to_dtype(dtype)?
            }
        } else {
            tensor.to_dtype(dtype)?
        };
        merged.insert(name.clone(), tensor);
    }

    println!("  Merged {merge_count} weight matrices with LoRA deltas.");
    Ok(VarBuilder::from_tensors(merged, dtype, device))
}

// ---------------------------------------------------------------------------
// Page image loading (image file or PDF)
// ---------------------------------------------------------------------------

fn load_page_images(args: &Args) -> Result<Vec<DynamicImage>> {
    #[cfg(feature = "pdf2image")]
    if let Some(ref pdf_path) = args.pdf {
        let pdf = PDF::from_file(pdf_path)
            .map_err(|e| anyhow::anyhow!("Failed to open PDF: {e}"))?;
        let page_count = pdf.page_count();
        println!("PDF: {pdf_path} ({page_count} pages)");

        let pages = if let Some(ref range_str) = args.pages {
            parse_page_range(range_str, page_count)?
        } else {
            pdf2image::Pages::Range(1..=page_count)
        };

        let render_opts = RenderOptionsBuilder::default().build()?;
        let images = pdf
            .render(pages, render_opts)
            .map_err(|e| anyhow::anyhow!("Failed to render PDF pages: {e}"))?;
        println!("Rendered {} page(s)", images.len());
        return Ok(images);
    }

    let image_path = args
        .image
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("Provide either --image or --pdf"))?;
    let img = image::ImageReader::open(image_path)?
        .decode()
        .map_err(E::msg)?;
    Ok(vec![img])
}

#[cfg(feature = "pdf2image")]
fn parse_page_range(s: &str, page_count: u32) -> Result<pdf2image::Pages> {
    if let Some((start, end)) = s.split_once('-') {
        let start: u32 = start.parse()?;
        let end: u32 = end.parse()?;
        if start < 1 || end > page_count || start > end {
            anyhow::bail!("Invalid page range {start}-{end} (PDF has {page_count} pages)");
        }
        Ok(pdf2image::Pages::Range(start..=end))
    } else {
        let page: u32 = s.parse()?;
        if page < 1 || page > page_count {
            anyhow::bail!("Invalid page {page} (PDF has {page_count} pages)");
        }
        Ok(pdf2image::Pages::Range(page..=page))
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let dtype = if args.bf16 { DType::BF16 } else { DType::F32 };

    let api = hf_hub::api::sync::Api::new()?;
    let model_id = args.model_id.as_deref().unwrap_or(MODEL_ID);
    let repo = api.model(model_id.to_string());

    // Load config
    let config_path = repo.get("config.json")?;
    let config: Config = serde_json::from_reader(std::fs::File::open(&config_path)?)?;

    println!(
        "Config: vision {}x{} patch={}, text hidden={} layers={}, downsample={}",
        config.vision_config.image_size,
        config.vision_config.patch_size,
        config.vision_config.num_hidden_layers,
        config.text_config.hidden_size,
        config.text_config.num_hidden_layers,
        config.downsample_rate,
    );
    println!(
        "  Deepstack: {:?}",
        config.deepstack_layer_map
    );
    println!(
        "  Spatial layers: {:?}",
        config.spatial_target_layers
    );

    // Load tokenizer
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

    // Load model weights (with automatic LoRA merging if adapter present)
    let weight_files = {
        let index_path = repo.get("model.safetensors.index.json")?;
        let index: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&index_path)?)?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("missing weight_map"))?;
        let mut files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        files.sort();
        files.dedup();
        files
            .into_iter()
            .map(|f| repo.get(&f))
            .collect::<std::result::Result<Vec<_>, _>>()?
    };

    // Check for LoRA adapter
    let adapter_path = repo.get("adapter_model.safetensors").ok();

    let vb = if let Some(ref adapter_path) = adapter_path {
        println!(
            "Loading model from {} shard(s) + LoRA adapter...",
            weight_files.len()
        );
        merge_lora_and_load(&weight_files, adapter_path, dtype, &device)?
    } else {
        println!(
            "Loading model from {} shard(s) (pre-merged, no LoRA)...",
            weight_files.len()
        );
        unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? }
    };

    let model = Model::new(&config, vb)?;
    println!("Model loaded.");

    // Load page images — either from a single image file or from PDF pages
    let page_images = load_page_images(&args)?;

    // Build prompt: expand task tag to full instruction (matching HF chat_template).
    // The \n prefix separates image tokens from instruction (matches Python template).
    let full_prompt = if let Some(ref custom) = args.prompt {
        format!("\n{custom}")
    } else {
        let expanded = expand_task_tag(&args.task);
        if expanded.is_empty() {
            format!("\n<{}>", args.task)
        } else {
            format!("\n{expanded}")
        }
    };

    let temperature = if args.temperature > 0.0 {
        Some(args.temperature)
    } else {
        None
    };

    let text_config = config.text_config.clone().into_config(false);

    // Process each page
    for (page_idx, page_img) in page_images.iter().enumerate() {
        if page_images.len() > 1 {
            println!("\n===== Page {} of {} =====", page_idx + 1, page_images.len());
        }

        let raw_img = page_img.to_rgb8();
        let preprocessed = preprocess_image(&raw_img, &config, args.no_split)?;
        let pixel_values = tiles_to_tensor(
            &preprocessed.tiles,
            config.vision_config.image_size,
            dtype,
            &device,
        )?;

        let input_ids = build_input_ids(
            &tokenizer,
            &full_prompt,
            preprocessed.num_image_tokens,
            config.image_token_index,
        )?;

        let n_image_tokens = input_ids
            .iter()
            .filter(|&&id| id == config.image_token_index)
            .count();
        println!(
            "Input: {} tokens ({} image, {} tiles)",
            input_ids.len(),
            n_image_tokens,
            pixel_values.dim(0)?,
        );

        let input_ids_tensor = Tensor::new(&input_ids[..], &device)?.unsqueeze(0)?;

        // Fresh KV cache per page
        let mut cache = GraniteMoeHybridCache::new(true, dtype, &text_config, &device)?;

        // Initial forward with vision
        let logits = model.setup(
            &input_ids_tensor,
            &pixel_values,
            preprocessed.image_size,
            &mut cache,
        )?;

        let mut logits_processor =
            candle_transformers::generation::LogitsProcessor::new(42, temperature, None);

        let logits_last = logits.squeeze(0)?.squeeze(0)?;
        let mut token = logits_processor.sample(&logits_last)?;
        let mut generated = vec![token];
        print_token(&tokenizer, token);

        // Autoregressive generation
        let index_pos = input_ids.len();
        for step in 1..args.max_tokens {
            let input = Tensor::new(&[token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, index_pos + step - 1, &mut cache)?;
            let logits_last = logits.squeeze(0)?;
            token = logits_processor.sample(&logits_last)?;

            if token == END_OF_TEXT {
                break;
            }
            generated.push(token);
            print_token(&tokenizer, token);
        }

        println!();
        println!("Generated {} tokens.", generated.len());
    }

    Ok(())
}

fn print_token(tokenizer: &Tokenizer, token: u32) {
    if let Ok(text) = tokenizer.decode(&[token], false) {
        use std::io::Write;
        print!("{text}");
        let _ = std::io::stdout().flush();
    }
}
