//! Depth Anything V2
//! https://huggingface.co/spaces/depth-anything/Depth-Anything-V2

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use clap::Parser;
use std::{ffi::OsString, path::PathBuf, sync::Arc};

use candle::DType::{F32, U8};
use candle::{DType, Device, Module, Result, Tensor};
use candle_examples::{load_image, load_image_and_resize, save_image};
use candle_nn::VarBuilder;
use candle_transformers::models::depth_anything_v2::{DepthAnythingV2, DepthAnythingV2Config};
use candle_transformers::models::dinov2;

use crate::color_map::SpectralRColormap;

mod color_map;

// taken these from: https://huggingface.co/spaces/depth-anything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py#L207
const MAGIC_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const MAGIC_STD: [f32; 3] = [0.229, 0.224, 0.225];

const DINO_IMG_SIZE: usize = 518;

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

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    color_map: bool,
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

    let config = DepthAnythingV2Config::vit_small();
    let depth_anything = DepthAnythingV2::new(Arc::new(dinov2), config, vb)?;

    let (original_height, original_width, image) = load_and_prep_image(&args.image, &device)?;

    println!("Loaded image {image:?}");

    let depth = depth_anything.forward(&image)?;

    println!("Got predictions {:?}", depth.shape());

    let output_image = post_process_image(&depth, original_height, original_width, args.color_map)?;

    let output_path = full_output_path(&args.image, &args.output_dir);
    println!("Saving image to {}", output_path.to_string_lossy());
    save_image(&output_image, output_path)?;

    Ok(())
}

fn full_output_path(image_path: &PathBuf, output_dir: &Option<PathBuf>) -> PathBuf {
    let input_file_name = image_path.file_name().unwrap();
    let mut output_file_name = OsString::from("depth_");
    output_file_name.push(input_file_name);
    let mut output_path = match output_dir {
        None => image_path.parent().unwrap().to_path_buf(),
        Some(output_path) => output_path.clone(),
    };
    output_path.push(output_file_name);

    output_path
}

fn load_and_prep_image(
    image_path: &PathBuf,
    device: &Device,
) -> anyhow::Result<(usize, usize, Tensor)> {
    let (_original_image, original_height, original_width) = load_image(&image_path, None)?;

    let image = load_image_and_resize(&image_path, DINO_IMG_SIZE, DINO_IMG_SIZE)?
        .unsqueeze(0)?
        .to_dtype(F32)?
        .to_device(&device)?;

    let max_pixel_val = Tensor::try_from(255.0f32)?
        .to_device(&device)?
        .broadcast_as(image.shape())?;
    let image = (image / max_pixel_val)?;
    let image = normalize_image(&image, &MAGIC_MEAN, &MAGIC_STD)?;

    Ok((original_height, original_width, image))
}

fn normalize_image(image: &Tensor, mean: &[f32; 3], std: &[f32; 3]) -> Result<Tensor> {
    let mean_tensor =
        Tensor::from_vec(mean.to_vec(), (3, 1, 1), &image.device())?.broadcast_as(image.shape())?;
    let std_tensor =
        Tensor::from_vec(std.to_vec(), (3, 1, 1), &image.device())?.broadcast_as(image.shape())?;
    image.sub(&mean_tensor)?.div(&std_tensor)
}

fn post_process_image(
    image: &Tensor,
    original_height: usize,
    original_width: usize,
    color_map: bool,
) -> Result<Tensor> {
    let out = image.interpolate2d(original_height, original_width)?;
    let out = scale_image(&out)?;

    let out = if color_map {
        let spectral_r = SpectralRColormap::new();
        spectral_r.gray2color(&out)?
    } else {
        let rgb_slice = [&out, &out, &out];
        Tensor::cat(&rgb_slice, 0)?.squeeze(1)?
    };

    let max_pixel_val = Tensor::try_from(255.0f32)?
        .to_device(out.device())?
        .broadcast_as(out.shape())?;
    let out = (out * max_pixel_val)?;

    out.to_dtype(U8)
}

fn scale_image(depth: &Tensor) -> Result<Tensor> {
    let flat_values: Vec<f32> = depth.flatten_all()?.to_vec1()?;

    let min_val = flat_values.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let max_val = flat_values.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    let min_val_tensor = Tensor::try_from(*min_val)?
        .to_device(depth.device())?
        .broadcast_as(depth.shape())?;
    let depth = (depth - min_val_tensor)?;

    let range = max_val - min_val;
    let range_tensor = Tensor::try_from(range)?
        .to_device(depth.device())?
        .broadcast_as(depth.shape())?;

    depth / range_tensor
}
