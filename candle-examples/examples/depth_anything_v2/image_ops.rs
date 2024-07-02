use std::cmp::Ordering;
use std::path::PathBuf;
use enterpolation::DiscreteGenerator;
use image::{DynamicImage, GenericImageView};

use candle::DType::{F32, F64, U8};
use candle::{Device, Tensor};
use candle_transformers::models::depth_anything_v2::PATCH_MULTIPLE;
use opencv::core::{flip, MatTraitConst, Point3_, Point3d, Rect, Scalar, Size};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::imgproc::resize;
use opencv::prelude::*;

use crate::color_map::SpectralRColormap;

// taken these from: https://huggingface.co/spaces/depth-anything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py#L207
const MAGIC_MEAN: [f64; 3] = [0.485, 0.456, 0.406];
const MAGIC_STD: [f64; 3] = [0.229, 0.224, 0.225];

const LOWER_BOUND: &'static str = "lower_bound";
const UPPER_BOUND: &'static str = "upper_bound";
const MINIMAL: &'static str = "minimal";
const DINO_IMG_SIZE: usize = 518;

fn print_tensor_statistics(tensor: &Tensor) -> candle::Result<()> {
    // General characteristics
    println!("Tensor: {:?}", tensor);

    let tensor = tensor.flatten_all().unwrap();

    // Inline method to compute the minimum value over all elements
    fn min_all(tensor: &Tensor) -> f32 {
        let vec: Vec<f32> = tensor.to_vec1().unwrap();
        vec.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap()
    }

    // Inline method to compute the maximum value over all elements
    fn max_all(tensor: &Tensor) -> f32 {
        let vec: Vec<f32> = tensor.to_vec1().unwrap();
        vec.iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap()
    }

    // Inline method to compute the mean value over all elements
    fn mean_all(tensor: &Tensor) -> candle::Result<f32> {
        let vec: Vec<f32> = tensor.to_vec1()?;
        Ok(vec.iter().sum::<f32>() / vec.len() as f32)
    }

    // Compute overall statistics
    let min_val = min_all(&tensor);
    let max_val = max_all(&tensor);
    let mean_val = mean_all(&tensor)?;

    println!("Overall statistics:");
    println!("  Min: {:?}", min_val);
    println!("  Max: {:?}", max_val);
    println!("  Mean: {:?}", mean_val);

    Ok(())
}

fn print_tensor_statistics_f64(tensor: &Tensor) -> candle::Result<()> {
    // General characteristics
    println!("Tensor: {:?}", tensor);

    let tensor = tensor.flatten_all().unwrap();

    // Inline method to compute the minimum value over all elements
    fn min_all(tensor: &Tensor) -> f64 {
        let vec: Vec<f64> = tensor.to_vec1().unwrap();
        vec.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap()
    }

    // Inline method to compute the maximum value over all elements
    fn max_all(tensor: &Tensor) -> f64 {
        let vec: Vec<f64> = tensor.to_vec1().unwrap();
        vec.iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap()
    }

    // Inline method to compute the mean value over all elements
    fn mean_all(tensor: &Tensor) -> candle::Result<f64> {
        let vec: Vec<f64> = tensor.to_vec1()?;
        Ok(vec.iter().sum::<f64>() / vec.len() as f64)
    }

    // Compute overall statistics
    let min_val = min_all(&tensor);
    let max_val = max_all(&tensor);
    let mean_val = mean_all(&tensor)?;

    println!("Overall statistics:");
    println!("  Min: {:?}", min_val);
    println!("  Max: {:?}", max_val);
    println!("  Mean: {:?}", mean_val);

    Ok(())
}

fn print_tensor_statistics_u8(tensor: &Tensor) -> candle::Result<()> {
    // General characteristics
    println!("Tensor: {:?}", tensor);

    let tensor = tensor.flatten_all().unwrap();

    // Inline method to compute the minimum value over all elements
    fn min_all(tensor: &Tensor) -> u8 {
        let vec: Vec<u8> = tensor.to_vec1().unwrap();
        vec.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap()
    }

    // Inline method to compute the maximum value over all elements
    fn max_all(tensor: &Tensor) -> u8 {
        let vec: Vec<u8> = tensor.to_vec1().unwrap();
        vec.iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap()
    }

    // Inline method to compute the mean value over all elements
    fn mean_all(tensor: &Tensor) -> candle::Result<f32> {
        let vec: Vec<u8> = tensor.to_vec1()?;
        let vec: Vec<u32> = vec.iter().map(|u| u as u32).collect();
        Ok(vec.iter().sum::<u32>() as f32 / vec.len() as f32)
    }

    // Compute overall statistics
    let min_val = min_all(&tensor);
    let max_val = max_all(&tensor);
    let mean_val = mean_all(&tensor)?;

    println!("Overall statistics:");
    println!("  Min: {:?}", min_val);
    println!("  Max: {:?}", max_val);
    println!("  Mean: {:?}", mean_val);

    Ok(())
}

fn print_image_statistics(image: DynamicImage) {
    // General characteristics
    println!("Image dimensions: {:?}", image.dimensions());
    println!("Color type: {:?}", image.color());

    // Flatten the image into a vector of u8 values
    let pixels: Vec<u8> = image.into_bytes();

    // Compute overall statistics
    let min_val = pixels.iter().min().unwrap();
    let max_val = pixels.iter().max().unwrap();
    let pixels: Vec<f64> = pixels.iter().map(|p| p as f64).collect();
    let mean_val = pixels.iter().sum::<f64>() / pixels.len() as f64;

    println!("Overall statistics:");
    println!("  Min: {:?}", min_val);
    println!("  Max: {:?}", max_val);
    println!("  Mean: {:?}", mean_val);
}

fn print_mat_statistics_u8(mat: &Mat) -> anyhow::Result<()> {
    // General characteristics
    println!("Mat: {:?}", mat);
    println!("Dimensions: {:?}", mat.size()?);
    println!("Channels: {}", mat.channels());

    // Flatten the Mat into a single vector of u8 values
    let mut min_val = u8::MAX as f64;
    let mut max_val = u8::MIN as f64;
    let mut sum_val = 0.0;
    let mut count = 0;

    let mat_data: Vec<u8> = mat
        .data_typed::<Point3_<u8>>()
        .iter()
        .flat_map(|pv| {
            let pv = pv.to_vec();

            pv.iter().flat_map(|p| p.to_vec3()).collect::<Vec<u8>>()
        })
        .collect();
    // let mat_data = mat_data.iter().map(|p| Vec::from([p.x, p.y.p.z])).collect();
    for pixel in mat_data.iter() {
        min_val = min_val.min(pixel as f64);
        max_val = max_val.max(pixel as f64);
        sum_val += pixel as f64;
        count += 1;
    }
    let mean_val = sum_val / count as f64;

    println!("Overall statistics:");
    println!("  Min: {:?}", min_val);
    println!("  Max: {:?}", max_val);
    println!("  Mean: {:?}", mean_val);

    Ok(())
}

fn print_mat_statistics_f64(mat: &Mat) -> anyhow::Result<()> {
    // General characteristics
    println!("Mat: {:?}", mat);
    println!("Dimensions: {:?}", mat.size()?);
    println!("Channels: {}", mat.channels());

    // Flatten the Mat into a single vector of f64 values
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    let mut sum_val = 0.0;
    let mut count = 0;

    let mat_data: Vec<f64> = mat
        .data_typed::<Point3d>()
        .iter()
        .flat_map(|pv| {
            let pv = pv.to_vec();

            pv.iter().flat_map(|p| p.to_vec3()).collect::<Vec<f64>>()
        })
        .collect();

    for pixel in mat_data.iter() {
        min_val = min_val.min(pixel as f64);
        max_val = max_val.max(pixel as f64);
        sum_val += pixel as f64;
        count += 1;
    }

    let mean_val = sum_val / count as f64;

    println!("Overall statistics:");
    println!("  Min: {:?}", min_val);
    println!("  Max: {:?}", max_val);
    println!("  Mean: {:?}", mean_val);

    Ok(())
}

fn print_mat_statistics(mat: &Mat) -> anyhow::Result<()> {
    match mat.typ() {
        opencv::core::CV_8UC1 | opencv::core::CV_8UC3 => print_mat_statistics_u8(mat),
        opencv::core::CV_64FC1 | opencv::core::CV_64FC3 => print_mat_statistics_f64(mat),
        _ => Err(anyhow::anyhow!("Unsupported Mat type")),
    }
}

pub fn load_and_prep_image(
    image_path: &PathBuf,
    device: &Device,
) -> anyhow::Result<(usize, usize, Tensor)> {
    println!("Loading image");
    let (image, original_height, original_width) = load_image_and_resize(&image_path)?;
    println!("Got image {image:?}, with original size ({original_height}, {original_width})");

    println!("Pre normalization");
    print_tensor_statistics_f64(&image)?;
    println!("Normalizing image");
    let image = normalize_image(&image, &MAGIC_MEAN, &MAGIC_STD)?;
    println!("Post normalization");
    print_tensor_statistics_f64(&image)?;

    println!("Pre net prep");
    print_tensor_statistics_f64(&image)?;
    let image = image.permute((2, 0, 1))?.unsqueeze(0)?;
    println!("Post net prep");
    print_tensor_statistics_f64(&image)?;

    let image = image.to_dtype(F32)?.to_device(device)?;
    println!("Post net prep");
    print_tensor_statistics(&image)?;

    Ok((original_height, original_width, image))
}

pub fn load_image_and_resize<P: AsRef<std::path::Path>>(
    p: P,
) -> anyhow::Result<(Tensor, usize, usize)> {
    // let img = image::io::Reader::open(p)?
    //     .decode()
    //     .map_err(candle::Error::wrap)?;
    //
    // println!("Raw image");
    // print_image_statistics(img.clone());
    let image = imgcodecs::imread(&p.as_ref().to_string_lossy(), imgcodecs::IMREAD_COLOR)?;

    // let (original_height, original_width) = (img.height() as usize, img.width() as usize);
    let size = image.size()?;
    let mut float_image = unsafe { Mat::new_size(image.size()?, opencv::core::CV_64FC3)? };
    image.convert_to(&mut float_image, opencv::core::CV_64FC3, 1.0 / 255.0, 0.0)?;

    let (target_height, target_width) = get_new_size(
        size.height as usize,
        size.width as usize,
        DINO_IMG_SIZE,
        DINO_IMG_SIZE,
        true,
        LOWER_BOUND,
        PATCH_MULTIPLE,
    );
    let mut source_image = Mat::default();
    println!("Pre Resize");
    print_mat_statistics(&float_image)?;
    resize(
        &float_image,
        &mut source_image,
        Size::new(target_width as i32, target_height as i32),
        0.0,
        0.0,
        imgproc::INTER_CUBIC,
    )?;
    println!("Post Resize");
    print_mat_statistics(&source_image)?;

    // let img = img.resize_to_fill(
    //     target_width as u32,
    //     target_height as u32,
    //     image::imageops::FilterType::CatmullRom,
    // );
    // println!("Post resize");
    // print_image_statistics(img.clone());
    //
    // let img = img.into_rgb32f();
    // let data = img.into_raw();
    let data: Vec<f64> = source_image
        .data_typed::<Point3d>()
        .iter()
        .flat_map(|pv| {
            let pv = pv.to_vec();

            pv.iter().flat_map(|p| p.to_vec3()).collect::<Vec<f64>>()
        })
        .collect();

    let bgr_data = Tensor::from_slice(
        &data,
        (target_height as usize, target_width as usize, 3),
        &Device::Cpu,
    )?;
    println!("BGR Image");
    print_tensor_statistics_f64(&bgr_data)?;

    // let index = Tensor::from_vec(vec![2u32, 1, 0], (3,), &Device::Cpu)?;
    // let  rgb_data = bgr_data.index_select(&index, 2)?;
    // println!("rgb Image");
    // print_tensor_statistics(&rgb_data)?;

    Ok((bgr_data, size.height as usize, size.width as usize))
}

fn normalize_image(image: &Tensor, mean: &[f64; 3], std: &[f64; 3]) -> candle::Result<Tensor> {
    let shape = (1, 1, 3);
    let mean_tensor = Tensor::from_vec(mean.to_vec(), shape, &image.device())?;
    let std_tensor = Tensor::from_vec(std.to_vec(), shape, &image.device())?;
    image
        .broadcast_sub(&mean_tensor)?
        .broadcast_div(&std_tensor)
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

pub fn get_min_max(t: &Tensor) -> candle::Result<(f32, f32)> {
    let flat_values: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let min_val = flat_values.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let max_val = flat_values.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    Ok((min_val, max_val))
}

fn scale_image(depth: &Tensor) -> candle::Result<Tensor> {
    let (min_val, max_val) = get_min_max(&depth)?;

    let min_val_tensor = Tensor::try_from(min_val)?
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
