//! BiRefNet: Bilateral Reference Network for High-Resolution Dichotomous Image Segmentation
//!
//! This example implements background removal using the BiRefNet model.
//!
//! BiRefNet is a state-of-the-art model for dichotomous image segmentation,
//! which can be used for high-quality background removal.
//!
//! - Paper: <https://arxiv.org/abs/2401.03407>
//! - GitHub: <https://github.com/ZhengPeng7/BiRefNet>
//! - HuggingFace: <https://huggingface.co/ZhengPeng7/BiRefNet>
//!
//! ## Usage
//!
//! ```bash
//! # Download model weights from HuggingFace automatically
//! cargo run --example birefnet --release -- \
//!     --image input.jpg \
//!     --output output.png
//!
//! # Use a local model file
//! cargo run --example birefnet --release -- \
//!     --model path/to/model.safetensors \
//!     --image input.jpg \
//!     --output output.png
//!
//! # Output mask only (grayscale)
//! cargo run --example birefnet --release -- \
//!     --image input.jpg \
//!     --output mask.png \
//!     --mask-only
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::birefnet::{BiRefNet, Config};
use clap::Parser;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Default HuggingFace model repository
const DEFAULT_MODEL_REPO: &str = "ZhengPeng7/BiRefNet";
/// Default model filename
const DEFAULT_MODEL_FILE: &str = "model.safetensors";

#[derive(Parser)]
#[command(about = "BiRefNet: Background removal using Bilateral Reference Network")]
struct Args {
    /// Path to the model weights file. If not provided, downloads from HuggingFace.
    #[arg(long)]
    model: Option<String>,

    /// HuggingFace model repository ID (default: ZhengPeng7/BiRefNet)
    #[arg(long, default_value = DEFAULT_MODEL_REPO)]
    model_id: String,

    /// Input image path
    #[arg(long)]
    image: String,

    /// Output image path
    #[arg(long, default_value = "birefnet_output.png")]
    output: String,

    /// Input size for the model (default: 1024)
    #[arg(long, default_value = "1024")]
    size: usize,

    /// Output grayscale mask only instead of RGBA image
    #[arg(long)]
    mask_only: bool,

    /// Run on CPU rather than GPU
    #[arg(long)]
    cpu: bool,

    /// Run N iterations for benchmarking (with warmup)
    #[arg(long, default_value = "0")]
    bench: usize,
}

/// Load and preprocess image for BiRefNet
fn load_and_preprocess_image(
    image_path: &str,
    target_size: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, u32, u32)> {
    let img = image::ImageReader::open(image_path)
        .map_err(candle::Error::wrap)?
        .decode()
        .map_err(candle::Error::wrap)?;

    let (orig_width, orig_height) = img.dimensions();

    let img = img.resize_exact(
        target_size as u32,
        target_size as u32,
        image::imageops::FilterType::Triangle,
    );

    let img = img.to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);

    let data: Vec<f32> = img
        .pixels()
        .flat_map(|p| {
            p.0.iter().enumerate().map(|(c, &v)| {
                let normalized = v as f32 / 255.0;
                (normalized - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            })
        })
        .collect();

    let tensor = Tensor::from_vec(data, (height, width, 3), device)?
        .permute((2, 0, 1))?
        .unsqueeze(0)?
        .to_dtype(dtype)?;

    Ok((tensor, orig_width, orig_height))
}

/// Apply sigmoid to tensor values
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Save output mask as grayscale image
fn save_mask_image(
    mask: &Tensor,
    output_path: &str,
    orig_width: u32,
    orig_height: u32,
) -> Result<()> {
    let (_, _, h, w) = mask.dims4()?;
    let mask = mask.squeeze(0)?.squeeze(0)?;
    let mask = mask.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let mask_data: Vec<f32> = mask.flatten_all()?.to_vec1()?;

    let mask_data: Vec<f32> = mask_data.iter().map(|&v| sigmoid(v)).collect();

    let mut img_buf: ImageBuffer<image::Luma<u8>, Vec<u8>> = ImageBuffer::new(w as u32, h as u32);
    for (i, pixel) in img_buf.pixels_mut().enumerate() {
        let v = (mask_data[i] * 255.0).clamp(0.0, 255.0) as u8;
        *pixel = image::Luma([v]);
    }

    let img = DynamicImage::ImageLuma8(img_buf);
    let img = img.resize_exact(
        orig_width,
        orig_height,
        image::imageops::FilterType::Triangle,
    );

    img.save(output_path).map_err(candle::Error::wrap)?;
    Ok(())
}

/// Save output as RGBA image with transparent background
fn save_rgba_image(input_path: &str, mask: &Tensor, output_path: &str) -> Result<()> {
    let orig_img = image::ImageReader::open(input_path)
        .map_err(candle::Error::wrap)?
        .decode()
        .map_err(candle::Error::wrap)?;

    let (orig_width, orig_height) = orig_img.dimensions();
    let orig_rgb = orig_img.to_rgb8();

    let (_, _, h, w) = mask.dims4()?;
    let mask = mask.squeeze(0)?.squeeze(0)?;
    let mask = mask.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let mask_data: Vec<f32> = mask.flatten_all()?.to_vec1()?;

    let mask_data: Vec<f32> = mask_data.iter().map(|&v| sigmoid(v)).collect();

    let mut mask_buf: ImageBuffer<image::Luma<u8>, Vec<u8>> = ImageBuffer::new(w as u32, h as u32);
    for (i, pixel) in mask_buf.pixels_mut().enumerate() {
        let v = (mask_data[i] * 255.0).clamp(0.0, 255.0) as u8;
        *pixel = image::Luma([v]);
    }

    let mask_img = DynamicImage::ImageLuma8(mask_buf);
    let mask_img = mask_img.resize_exact(
        orig_width,
        orig_height,
        image::imageops::FilterType::Triangle,
    );
    let mask_gray = mask_img.to_luma8();

    let mut rgba_buf: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(orig_width, orig_height);
    for (x, y, pixel) in rgba_buf.enumerate_pixels_mut() {
        let rgb = orig_rgb.get_pixel(x, y);
        let alpha = mask_gray.get_pixel(x, y).0[0];
        *pixel = Rgba([rgb.0[0], rgb.0[1], rgb.0[2], alpha]);
    }

    rgba_buf.save(output_path).map_err(candle::Error::wrap)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("BiRefNet Background Removal");
    println!("===========================");

    let device = candle_examples::device(args.cpu)?;
    println!("Device: {:?}", device);

    // BiRefNet uses F32 precision (model weights are stored as F32)
    let dtype = DType::F32;
    println!("Dtype: {:?}", dtype);

    // Load model
    println!("\nLoading model...");
    let model_path = match &args.model {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let repo = api.model(args.model_id.clone());
            repo.get(DEFAULT_MODEL_FILE)?
        }
    };
    println!("Model: {:?}", model_path);

    let config = Config::default();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&model_path], dtype, &device)? };
    let model = BiRefNet::new(config, vb)?;
    println!("Model loaded successfully");

    // Load and preprocess image
    println!("\nProcessing image: {}", args.image);
    let start = std::time::Instant::now();
    let (input_tensor, orig_width, orig_height) =
        load_and_preprocess_image(&args.image, args.size, &device, dtype)?;
    println!(
        "Original size: {}x{}, preprocessed in {:.2?}",
        orig_width,
        orig_height,
        start.elapsed()
    );

    // Run inference
    println!("\nRunning inference...");
    let start = std::time::Instant::now();
    let output = model.forward(&input_tensor)?;
    // Force GPU sync
    let _ = output.flatten_all()?.get(0)?.to_scalar::<f32>()?;
    println!("Inference completed in {:.2?}", start.elapsed());

    // Benchmark mode
    if args.bench > 0 {
        println!("\nBenchmarking ({} iterations)...", args.bench);
        let mut times = Vec::with_capacity(args.bench);

        // Warmup
        for _ in 0..3 {
            let out = model.forward(&input_tensor)?;
            let _ = out.flatten_all()?.get(0)?.to_scalar::<f32>()?;
        }

        // Benchmark
        for i in 0..args.bench {
            let start = std::time::Instant::now();
            let out = model.forward(&input_tensor)?;
            let _ = out.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            let elapsed = start.elapsed();
            times.push(elapsed.as_secs_f64() * 1000.0);
            if (i + 1) % 10 == 0 {
                println!(
                    "  [{}/{}] {:.2}ms",
                    i + 1,
                    args.bench,
                    times.last().unwrap()
                );
            }
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = times.first().unwrap();
        let max = times.last().unwrap();
        let avg: f64 = times.iter().sum::<f64>() / times.len() as f64;
        let median = times[times.len() / 2];
        let p95 = times[(times.len() as f64 * 0.95) as usize];

        println!("\nBenchmark Results:");
        println!("  Min:    {:.2}ms", min);
        println!("  Max:    {:.2}ms", max);
        println!("  Avg:    {:.2}ms", avg);
        println!("  Median: {:.2}ms", median);
        println!("  P95:    {:.2}ms", p95);
        println!("  FPS:    {:.2}", 1000.0 / avg);
    }

    // Save output
    println!("\nSaving output to: {}", args.output);
    if args.mask_only {
        save_mask_image(&output, &args.output, orig_width, orig_height)?;
        println!("Mask saved successfully");
    } else {
        save_rgba_image(&args.image, &output, &args.output)?;
        println!("RGBA image saved successfully");
    }

    println!("\nDone!");
    Ok(())
}
