use std::path::PathBuf;

use candle::DType::{F32, U8};
use candle::{Device, Tensor};
use candle_examples::{load_image, load_image_and_resize};
use candle_transformers::models::depth_anything_v2::PATCH_MULTIPLE;

use crate::color_map::SpectralRColormap;

// taken these from: https://huggingface.co/spaces/depth-anything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py#L207
const MAGIC_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const MAGIC_STD: [f32; 3] = [0.229, 0.224, 0.225];

const LOWER_BOUND: &'static str = "lower_bound";
const UPPER_BOUND: &'static str = "upper_bound";
const MINIMAL: &'static str = "minimal";
const DINO_IMG_SIZE: usize = 518;

pub fn load_and_prep_image(
    image_path: &PathBuf,
    device: &Device,
) -> anyhow::Result<(usize, usize, Tensor)> {
    let (_original_image, original_height, original_width) = load_image(&image_path, None)?;

    let (target_height, target_width) = get_new_size(
        original_height,
        original_width,
        DINO_IMG_SIZE,
        DINO_IMG_SIZE,
        true,
        LOWER_BOUND,
        PATCH_MULTIPLE,
    );
    let image = load_image_and_resize(&image_path, target_width, target_height)?
        .permute((0, 2, 1))? // see issue #2291
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

fn normalize_image(image: &Tensor, mean: &[f32; 3], std: &[f32; 3]) -> candle::Result<Tensor> {
    let mean_tensor =
        Tensor::from_vec(mean.to_vec(), (3, 1, 1), &image.device())?.broadcast_as(image.shape())?;
    let std_tensor =
        Tensor::from_vec(std.to_vec(), (3, 1, 1), &image.device())?.broadcast_as(image.shape())?;
    image.sub(&mean_tensor)?.div(&std_tensor)
}

pub fn post_process_image(
    image: &Tensor,
    original_height: usize,
    original_width: usize,
    color_map: bool,
) -> candle::Result<Tensor> {
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

fn scale_image(depth: &Tensor) -> candle::Result<Tensor> {
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

fn constrain_to_multiple_of(
    value: f32,
    multiple_of: usize,
    min_val: f32,
    max_val: Option<f32>,
) -> usize {
    let mut constrained_value = (value / multiple_of as f32).round() * multiple_of as f32;

    if let Some(max_val) = max_val {
        if constrained_value > max_val {
            constrained_value = (value / multiple_of as f32).floor() * multiple_of as f32;
        }
    }

    if constrained_value < min_val {
        constrained_value = (value / multiple_of as f32).ceil() * multiple_of as f32;
    }

    constrained_value as usize
}

fn get_new_size(
    height: usize,
    width: usize,
    target_height: usize,
    target_width: usize,
    keep_aspect_ratio: bool,
    resize_method: &str,
    multiple_of: usize,
) -> (usize, usize) {
    let scale_width = target_width as f32 / width as f32;
    let scale_height = target_height as f32 / height as f32;

    let (new_width, new_height) = if keep_aspect_ratio {
        match resize_method {
            LOWER_BOUND => {
                if scale_width > scale_height {
                    (target_width, (scale_width * height as f32).round() as usize)
                } else {
                    (
                        (scale_height * width as f32).round() as usize,
                        target_height,
                    )
                }
            }
            UPPER_BOUND => {
                if scale_width < scale_height {
                    (target_width, (scale_width * height as f32).round() as usize)
                } else {
                    (
                        (scale_height * width as f32).round() as usize,
                        target_height,
                    )
                }
            }
            MINIMAL => {
                if (1.0 - scale_width).abs() < (1.0 - scale_height).abs() {
                    (target_width, (scale_width * height as f32).round() as usize)
                } else {
                    (
                        (scale_height * width as f32).round() as usize,
                        target_height,
                    )
                }
            }
            _ => panic!("resize_method {} not implemented", resize_method),
        }
    } else {
        (target_width, target_height)
    };

    let new_height =
        constrain_to_multiple_of(new_height as f32, multiple_of, target_height as f32, None);
    let new_width =
        constrain_to_multiple_of(new_width as f32, multiple_of, target_width as f32, None);

    (new_height, new_width) // switching height and width because the tensor is also height first
}
