//! Depth Anything V2
//! https://huggingface.co/spaces/depth-anything/Depth-Anything-V2

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::ffi::OsString;
use std::path::PathBuf;

use clap::Parser;

use args::Args;
use candle::{Device, Module};
use candle::DType::F32;
use candle_examples::save_image;
use candle_nn::VarBuilder;
use candle_transformers::models::depth_anything_v2::{DepthAnythingV2, DepthAnythingV2Config};
use candle_transformers::models::dinov2;
use candle_transformers::models::dinov2::DinoVisionTransformer;

use crate::args::ModelSize;

mod args;
mod color_map;
mod image_ops;

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;

    let dinov2 = dino_model(&args.size, args.dinov2_model, args.dinov2_head, &device)?;
    println!("DinoV2 model built");

    let depth_anything =
        depth_anything_model(&dinov2, &args.size, args.depth_anything_v2_model, &device)?;
    println!("Depth Anything model built");

    let (original_height, original_width, image) =
        image_ops::load_and_prep_image(&args.image, &device)?;

    println!("Loaded image {image:?}");

    let depth = depth_anything.forward(&image)?;

    println!("Got predictions {:?}", depth.shape());

    let output_image =
        image_ops::post_process_image(&depth, original_height, original_width, args.color_map)?;

    let output_path = full_output_path(&args.image, &args.output_dir);
    println!("Saving image to {}", output_path.to_string_lossy());
    save_image(&output_image, output_path)?;

    Ok(())
}

fn depth_anything_model<'a>(
    dinov2: &'a DinoVisionTransformer,
    model_size: &ModelSize,
    depth_anything_v2_model: Option<PathBuf>,
    device: &'a Device,
) -> anyhow::Result<DepthAnythingV2<'a>> {
    let api = hf_hub::api::sync::Api::new()?;
    let depth_anything_path = match depth_anything_v2_model {
        None => match model_size {
            ModelSize::S => api
                .model("jeroenvlek/depth-anything-v2-safetensors".into())
                .get("depth_anything_v2_vits.safetensors")?,
            ModelSize::B => api
                .model("jeroenvlek/depth-anything-v2-safetensors".into())
                .get("depth_anything_v2_vitb.safetensors")?,
            ModelSize::L => api
                .model("jeroenvlek/depth-anything-v2-safetensors".into())
                .get("depth_anything_v2_vitl.safetensors")?,
            ModelSize::G => todo!("At the time of writing the giant model isn't released yet"),
        },
        Some(depth_anything_model) => depth_anything_model,
    };
    println!("Using file {:?}", depth_anything_path);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[depth_anything_path], F32, &device)? };
    let config = match model_size {
        ModelSize::S => DepthAnythingV2Config::vit_small(),
        ModelSize::B => DepthAnythingV2Config::vit_base(),
        ModelSize::L => DepthAnythingV2Config::vit_large(),
        ModelSize::G => DepthAnythingV2Config::vit_giant(),
    };

    Ok(DepthAnythingV2::new(dinov2, config, vb)?)
}

fn dino_model(
    model_size: &ModelSize,
    dinov2_model: Option<PathBuf>,
    dinov2_head: Option<PathBuf>,
    device: &Device,
) -> anyhow::Result<DinoVisionTransformer> {
    let api = hf_hub::api::sync::Api::new()?;
    let model_path = match dinov2_model {
        None => match model_size {
            ModelSize::S => api
                .model("facebook/dinov2-small".into())
                .get("model.safetensors")?,
            ModelSize::B => api
                .model("facebook/dinov2-base".into())
                .get("model.safetensors")?,

            ModelSize::L => api
                .model("facebook/dinov2-large".into())
                .get("model.safetensors")?,
            ModelSize::G => api
                .model("facebook/dinov2-giant".into())
                .get("model.safetensors")?,
        },
        Some(path) => path,
    };
    println!("Using dinov2 file {:?}", model_path);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], F32, &device)? };

    let head_path = match dinov2_head {
        None => match model_size {
            ModelSize::S => api
                .model("jeroenvlek/dinov2-linear-heads-safetensors".into())
                .get("dinov2_vits14_linear_head.safetensors")?,
            ModelSize::B => api
                .model("jeroenvlek/dinov2-linear-heads-safetensors".into())
                .get("dinov2_vitb14_linear_head.safetensors")?,
            ModelSize::L => api
                .model("jeroenvlek/dinov2-linear-heads-safetensors".into())
                .get("dinov2_vitl14_linear_head.safetensors")?,
            ModelSize::G => api
                .model("jeroenvlek/dinov2-linear-heads-safetensors".into())
                .get("dinov2_vitg14_linear_head.safetensors")?,
        },
        Some(path) => path,
    };
    println!("Using dinov2 head file {:?}", head_path);
    let vb_head = unsafe { VarBuilder::from_mmaped_safetensors(&[head_path], F32, &device)? };

    let model = match model_size {
        ModelSize::S => dinov2::vit_small(vb, Some(vb_head))?,
        ModelSize::B => dinov2::vit_base(vb, Some(vb_head))?,
        ModelSize::L => dinov2::vit_large(vb, Some(vb_head))?,
        ModelSize::G => dinov2::vit_giant(vb, Some(vb_head))?,
    };
    Ok(model)
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
