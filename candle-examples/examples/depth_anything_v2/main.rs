//! Depth Anything V2
//! https://huggingface.co/spaces/depth-anything/Depth-Anything-V2

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use clap::Parser;
use std::ffi::OsString;
use std::path::PathBuf;

use candle::DType::{F32, U8};
use candle::{DType, Module, Result, Tensor};
use candle_examples::{load_image, load_image_and_resize, save_image};
use candle_nn::VarBuilder;
use candle_transformers::models::depth_anything_v2::{DepthAnythingV2, DINO_IMG_SIZE};
use candle_transformers::models::dinov2;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    dinov2_model: Option<PathBuf>,

    #[arg(long)]
    depth_anything_v2_model: Option<PathBuf>,

    #[arg(long)]
    image: PathBuf,

    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    let dinov2_model_file = match args.dinov2_model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/candle-dino-v2".into());
            api.get("dinov2_vits14.safetensors")?
        }
        Some(dinov2_model) => dinov2_model,
    };
    println!("Using file {:?}", dinov2_model_file);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dinov2_model_file], F32, &device)? };
    let dinov2 = dinov2::vit_small(vb)?;
    println!("DinoV2 model built");

    let depth_anything_model_file = match args.depth_anything_v2_model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("jeroenvlek/depth-anything-v2-safetensors".into());
            api.get("depth_anything_v2_vits.safetensors")?
        }
        Some(depth_anything_model) => depth_anything_model,
    };
    println!("Using file {:?}", depth_anything_model_file);

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[depth_anything_model_file], DType::F32, &device)?
    };
    let out_channel_sizes = vec![48, 96, 192, 384];
    const IN_CHANNEL_SIZE: usize = 384;
    const NUM_FEATURES: usize = 64;
    let depth_anything = DepthAnythingV2::new(
        &dinov2,
        IN_CHANNEL_SIZE,
        out_channel_sizes,
        NUM_FEATURES,
        vb,
    )?;

    let (_original_image, original_height, original_width) = load_image(&args.image, None)?;

    let image = load_image_and_resize(&args.image, DINO_IMG_SIZE, DINO_IMG_SIZE)?
        .unsqueeze(0)?
        .to_dtype(F32)?
        .to_device(&device)?;
    println!("Loaded image {image:?}");

    let depth = depth_anything.forward(&image)?;

    println!("Got predictions {:?}", depth.shape());

    let depth = post_process_image(&depth, original_height, original_width)?;

    let input_file_name = args.image.file_name().unwrap();
    let mut output_file_name = OsString::from("depth_");
    output_file_name.push(input_file_name);
    let mut output_path = match args.output_dir {
        None => args.image.parent().unwrap().to_path_buf(),
        Some(output_path) => output_path,
    };
    output_path.push(output_file_name);
    println!("Saving image to {}", output_path.to_string_lossy());

    save_image(&depth, output_path)?;

    Ok(())
}

fn post_process_image(
    image: &Tensor,
    original_height: usize,
    original_width: usize,
) -> Result<Tensor> {
    let out = image.interpolate2d(original_height, original_width)?;
    let out = normalize_and_scale(&out)?;
    let out = out.squeeze(1)?;
    let out = out.to_dtype(U8)?;

    let rgb_slice = [&out, &out, &out];
    Tensor::cat(&rgb_slice, 0)
}

fn normalize_and_scale(depth: &Tensor) -> Result<Tensor> {
    let flat_values: Vec<f32> = depth.flatten_all()?.to_vec1()?;

    let min_val = flat_values.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let max_val = flat_values.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    println!("Min: {min_val}");
    println!("Max: {max_val}");

    let min_val_tensor = Tensor::try_from(*min_val)?
        .to_device(depth.device())?
        .broadcast_as(depth.shape())?;
    let depth = (depth - min_val_tensor)?;

    let range = max_val - min_val;
    let range_tensor = Tensor::try_from(range)?
        .to_device(depth.device())?
        .broadcast_as(depth.shape())?;
    let depth = (depth / range_tensor)?;

    let max_pixel_val = Tensor::try_from(255.0f32)?
        .to_device(depth.device())?
        .broadcast_as(depth.shape())?;
    depth * max_pixel_val
}
