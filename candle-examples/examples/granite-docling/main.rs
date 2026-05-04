#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::granite_docling::{
    config::{Config, QuantizedVisionConfig, TextConfig},
    quantized::Model as QuantizedModel,
    Model,
};
use image::DynamicImage;
#[cfg(feature = "pdf2image")]
use pdf2image::{RenderOptionsBuilder, PDF};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "ibm-granite/granite-docling-258M";
const QUANTIZED_MODEL_ID: &str = "ggml-org/granite-docling-258M-GGUF";
const IMAGE_TOKEN_ID: u32 = 100270;
const FAKE_TOKEN_AROUND_IMAGE: u32 = 100339;
const GLOBAL_IMG: u32 = 100340;
const START_OF_ROLE: u32 = 100264;
const END_OF_ROLE: u32 = 100265;
const END_OF_TEXT: u32 = 100257;

// row,col grid token ids; vocab positions are non-contiguous.
const ROW_COL_TOKENS: [[u32; 4]; 4] = [
    [100258, 100259, 100261, 100262],
    [100263, 100267, 100268, 100341],
    [100342, 100343, 100344, 100345],
    [100346, 100347, 100348, 100349],
];

#[derive(Parser, Debug)]
#[command(about = "Granite-Docling: document image to structured markup")]
struct Args {
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    bf16: bool,

    /// Image file (PNG/JPG).
    #[cfg_attr(not(feature = "pdf2image"), arg(long))]
    #[cfg_attr(feature = "pdf2image", arg(long, required_unless_present = "pdf"))]
    image: Option<String>,

    /// PDF file (requires the `pdf2image` feature and system poppler).
    #[cfg(feature = "pdf2image")]
    #[arg(long, required_unless_present = "image")]
    pdf: Option<String>,

    /// PDF page range (e.g. "7-8" or "7"). Defaults to all pages.
    #[cfg(feature = "pdf2image")]
    #[arg(long)]
    pages: Option<String>,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,

    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    no_split: bool,

    #[arg(long)]
    quantized: bool,
}

fn resize_and_normalize(img: &image::RgbImage, w: u32, h: u32) -> Vec<f32> {
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

fn split_image(
    img: &image::RgbImage,
    tile_size: usize,
    max_long_edge: usize,
) -> (usize, usize, Vec<Vec<f32>>) {
    let (orig_w, orig_h) = (img.width() as f64, img.height() as f64);
    let tile = tile_size as f64;

    let longest = orig_w.max(orig_h);
    let scale = (max_long_edge as f64) / longest;
    let scaled_w = (orig_w * scale).round();
    let scaled_h = (orig_h * scale).round();

    let n_cols = (scaled_w / tile).ceil() as usize;
    let n_rows = (scaled_h / tile).ceil() as usize;

    let grid_w = (n_cols * tile_size) as u32;
    let grid_h = (n_rows * tile_size) as u32;
    let resized =
        image::imageops::resize(img, grid_w, grid_h, image::imageops::FilterType::Lanczos3);

    println!(
        "Image splitting: {}x{} -> {}x{} grid ({}x{} tiles = {} + 1 global)",
        img.width(),
        img.height(),
        grid_w,
        grid_h,
        n_rows,
        n_cols,
        n_rows * n_cols,
    );

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

    let global = resize_and_normalize(img, ts, ts);
    tiles.push(global);

    (n_rows, n_cols, tiles)
}

fn load_single(img: &image::RgbImage, tile_size: usize) -> Vec<Vec<f32>> {
    println!(
        "No splitting: {}x{} -> {}x{}",
        img.width(),
        img.height(),
        tile_size,
        tile_size,
    );
    vec![resize_and_normalize(
        img,
        tile_size as u32,
        tile_size as u32,
    )]
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

    if n_rows > 4 || n_cols > 4 {
        anyhow::bail!(
            "Image splits into {n_rows}x{n_cols} grid, but max supported is 4x4. \
             Use a lower resolution image or --no-split."
        );
    }

    let mut ids: Vec<u32> = Vec::new();

    ids.push(START_OF_ROLE);
    ids.extend_from_slice(user_text.get_ids());
    ids.push(END_OF_ROLE);

    for row_tokens in ROW_COL_TOKENS.iter().take(n_rows) {
        for &token_id in row_tokens.iter().take(n_cols) {
            ids.push(FAKE_TOKEN_AROUND_IMAGE);
            ids.push(token_id);
            ids.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, image_seq_len));
        }
        ids.extend_from_slice(newline_enc.get_ids());
    }

    ids.extend_from_slice(newline_enc.get_ids());
    ids.push(FAKE_TOKEN_AROUND_IMAGE);
    ids.push(GLOBAL_IMG);
    ids.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, image_seq_len));
    ids.push(FAKE_TOKEN_AROUND_IMAGE);

    ids.extend_from_slice(prompt_enc.get_ids());

    ids.push(END_OF_TEXT);
    ids.extend_from_slice(newline_enc.get_ids());

    ids.push(START_OF_ROLE);
    ids.extend_from_slice(assistant_text.get_ids());
    ids.push(END_OF_ROLE);

    Ok(ids)
}

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
    ids.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, image_seq_len));
    ids.push(FAKE_TOKEN_AROUND_IMAGE);
    ids.extend_from_slice(prompt_enc.get_ids());
    ids.push(END_OF_TEXT);
    ids.extend_from_slice(newline_enc.get_ids());
    ids.push(START_OF_ROLE);
    ids.extend_from_slice(assistant_text.get_ids());
    ids.push(END_OF_ROLE);
    Ok(ids)
}

enum ModelWrapper {
    F32(Model),
    Quantized(QuantizedModel),
}

impl ModelWrapper {
    fn setup(&mut self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::F32(m) => Ok(m.setup(pixel_values, input_ids)?),
            Self::Quantized(m) => Ok(m.setup(pixel_values, input_ids)?),
        }
    }

    fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::F32(m) => Ok(m.forward(input_ids)?),
            Self::Quantized(m) => Ok(m.forward(input_ids)?),
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            Self::F32(m) => m.clear_kv_cache(),
            Self::Quantized(m) => m.clear_kv_cache(),
        }
    }
}

fn load_page_images(args: &Args) -> Result<Vec<DynamicImage>> {
    #[cfg(feature = "pdf2image")]
    if let Some(ref pdf_path) = args.pdf {
        let pdf =
            PDF::from_file(pdf_path).map_err(|e| anyhow::anyhow!("Failed to open PDF: {e}"))?;
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

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let api = hf_hub::api::sync::Api::new()?;

    let base_repo = api.model(MODEL_ID.to_string());
    let tokenizer_path = base_repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

    let (mut model, tile_size, image_seq_len) = if args.quantized {
        let model_id = args.model_id.as_deref().unwrap_or(QUANTIZED_MODEL_ID);
        let repo = api.model(model_id.to_string());

        let text_path = repo.get("granite-docling-258M-Q8_0.gguf")?;
        let vision_path = repo.get("mmproj-granite-docling-258M-Q8_0.gguf")?;

        println!("Loading quantized model...");
        println!("  Text:   {text_path:?}");
        println!("  Vision: {vision_path:?}");

        use candle::quantized::gguf_file;
        use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;

        let (text_vb, text_cfg) = {
            let mut f = std::fs::File::open(&text_path)?;
            let content = gguf_file::Content::read(&mut f)?;
            let cfg = TextConfig::from_gguf(&content.metadata)?;
            let vb = QVarBuilder::from_gguf(&text_path, &device)?;
            (vb, cfg)
        };
        let (vision_vb, vision_cfg) = {
            let mut f = std::fs::File::open(&vision_path)?;
            let content = gguf_file::Content::read(&mut f)?;
            let cfg = QuantizedVisionConfig::from_gguf(&content.metadata)?;
            let vb = QVarBuilder::from_gguf(&vision_path, &device)?;
            (vb, cfg)
        };

        let tile_size = vision_cfg.image_size;
        let image_seq_len = vision_cfg.image_seq_len();

        println!(
            "Config: vision {}x{} patch={}, text hidden={} layers={}, scale_factor={}",
            tile_size,
            tile_size,
            vision_cfg.patch_size,
            text_cfg.hidden_size,
            text_cfg.num_hidden_layers,
            vision_cfg.scale_factor,
        );

        let model =
            QuantizedModel::new(vision_vb, &vision_cfg, text_vb, &text_cfg, IMAGE_TOKEN_ID)?;
        println!("Model loaded (quantized Q8_0).");

        (ModelWrapper::Quantized(model), tile_size, image_seq_len)
    } else {
        let dtype = if args.bf16 { DType::BF16 } else { DType::F32 };
        let model_id = args.model_id.as_deref().unwrap_or(MODEL_ID);
        let repo = api.model(model_id.to_string());

        let config_path = repo.get("config.json")?;
        let config: Config = serde_json::from_reader(std::fs::File::open(&config_path)?)?;
        let tile_size = config.vision_config.image_size;
        let image_seq_len = config.image_seq_len();

        println!(
            "Config: vision {}x{} patch={}, text hidden={} layers={}, scale_factor={}",
            tile_size,
            tile_size,
            config.vision_config.patch_size,
            config.text_config.hidden_size,
            config.text_config.num_hidden_layers,
            config.scale_factor,
        );

        let model_path = repo.get("model.safetensors")?;
        println!("Loading model from {model_path:?}");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)? };
        let model = Model::new(&config, vb)?;
        println!("Model loaded (f32).");

        (ModelWrapper::F32(model), tile_size, image_seq_len)
    };

    let page_images: Vec<DynamicImage> = load_page_images(&args)?;

    let prompt_text = args
        .prompt
        .as_deref()
        .unwrap_or("Convert this page to docling.");
    let pv_dtype = if args.quantized {
        DType::F32
    } else if args.bf16 {
        DType::BF16
    } else {
        DType::F32
    };

    let temperature = if args.temperature > 0.0 {
        Some(args.temperature)
    } else {
        None
    };

    for (page_idx, page_img) in page_images.iter().enumerate() {
        if page_idx > 0 {
            model.clear_kv_cache();
        }
        if page_images.len() > 1 {
            println!(
                "\n===== Page {} of {} =====",
                page_idx + 1,
                page_images.len()
            );
        }

        let raw_img = page_img.to_rgb8();

        let (pixel_values, input_ids) = if args.no_split {
            let tiles = load_single(&raw_img, tile_size);
            let pv = tiles_to_tensor(&tiles, tile_size, pv_dtype, &device)?;
            let ids = build_input_ids_single(&tokenizer, prompt_text, image_seq_len)?;
            (pv, ids)
        } else {
            let (n_rows, n_cols, tiles) = split_image(&raw_img, tile_size, 2048);
            let pv = tiles_to_tensor(&tiles, tile_size, pv_dtype, &device)?;
            let ids =
                build_input_ids_split(&tokenizer, prompt_text, image_seq_len, n_rows, n_cols)?;
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

        let logits = model.setup(&pixel_values, &input_ids_tensor)?;
        let mut logits_processor =
            candle_transformers::generation::LogitsProcessor::new(42, temperature, None);

        let logits_last = logits.squeeze(0)?.get(logits.dim(1)? - 1)?;

        let mut token = logits_processor.sample(&logits_last)?;
        let mut generated = vec![token];
        print_token(&tokenizer, token);

        for _ in 1..args.max_tokens {
            let input = Tensor::new(&[token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input)?;
            let logits_last = logits.squeeze(0)?.get(logits.dim(1)? - 1)?;
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
