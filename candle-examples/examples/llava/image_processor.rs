use std::cmp::min;

use candle::{bail, DType, Device, Result, Tensor};
use candle_transformers::models::llava::{
    config::{HFPreProcessorConfig, LLaVAConfig},
    utils::select_best_resolution,
};
use hf_hub::api::sync::Api;
use image::{imageops::overlay, DynamicImage, GenericImageView, Rgb, RgbImage};
use serde::{Deserialize, Serialize};

//This struct is mainly for LLaVA aplications, hence it's not completely compatible with python transformer CLIPImageProcessor  few several preprocess that LLaVA used, including "openai/clip-vit-large-patch14-336" and "openai/clip-vit-large-patch14".

#[derive(Serialize, Deserialize, Debug)]
pub struct ImageProcessor {
    #[serde(default = "default_size")]
    pub size: u32, // this is not the same as python transformer
    #[serde(default = "default_do_resize")]
    pub do_resize: bool,

    //resample: u32 // 3 for PIL bicubic, equivalent to rust  CatmullRom. Hence below we use CatmullRom
    #[serde(default = "default_do_center_crop")]
    pub do_center_crop: bool,
    #[serde(default = "default_crop_size")]
    pub crop_size: u32, // this is not the same as python transformer
    #[serde(default = "default_do_rescale")]
    pub do_rescale: bool,
    #[serde(default = "default_rescale_factor")]
    pub rescale_factor: f32,
    #[serde(default = "default_do_normalize")]
    pub do_normalize: bool,
    #[serde(default = "default_image_mean")]
    pub image_mean: Vec<f32>,
    #[serde(default = "default_image_std")]
    pub image_std: Vec<f32>,
}

fn default_size() -> u32 {
    224
}

fn default_do_resize() -> bool {
    true
}

fn default_do_center_crop() -> bool {
    true
}

fn default_crop_size() -> u32 {
    224
}

fn default_do_rescale() -> bool {
    true
}

fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}

fn default_do_normalize() -> bool {
    true
}

fn default_image_mean() -> Vec<f32> {
    vec![0.48145466, 0.4578275, 0.40821073]
}

fn default_image_std() -> Vec<f32> {
    vec![0.26862954, 0.2613026, 0.2757771]
}

impl ImageProcessor {
    pub fn from_pretrained(clip_id: &str) -> Result<Self> {
        let api = Api::new().map_err(|e| candle::Error::Msg(e.to_string()))?;
        let api = api.model(clip_id.to_string());
        let config_filename = api
            .get("preprocessor_config.json")
            .map_err(|e| candle::Error::Msg(e.to_string()))?;
        let image_processor =
            serde_json::from_slice(&std::fs::read(config_filename).map_err(candle::Error::Io)?)
                .map_err(|e| candle::Error::Msg(e.to_string()))?;
        Ok(image_processor)
    }

    pub fn from_hf_preprocessor_config(hf_preprocessor_config: &HFPreProcessorConfig) -> Self {
        Self {
            size: hf_preprocessor_config.size["shortest_edge"] as u32,
            do_resize: hf_preprocessor_config.do_resize,
            do_center_crop: hf_preprocessor_config.do_center_crop,
            crop_size: hf_preprocessor_config.crop_size["height"] as u32,
            do_rescale: hf_preprocessor_config.do_rescale,
            rescale_factor: hf_preprocessor_config.rescale_factor,
            do_normalize: hf_preprocessor_config.do_normalize,
            image_mean: hf_preprocessor_config.image_mean.clone(),
            image_std: hf_preprocessor_config.image_std.clone(),
        }
    }

    ///shortest edge to self.resize, other edge is resized to maintain aspect ratio
    pub fn resize(&self, image: &DynamicImage) -> DynamicImage {
        let (width, height) = image.dimensions();
        let size = self.size;
        if width == size && height == size {
            image.clone()
        } else {
            let (new_width, new_height) = if width < height {
                (
                    size,
                    (((size * height) as f32) / width as f32).ceil() as u32,
                )
            } else {
                (
                    (((size * width) as f32) / height as f32).ceil() as u32,
                    size,
                )
            };
            image.resize(
                new_width,
                new_height,
                image::imageops::FilterType::CatmullRom,
            )
        }
    }

    pub fn center_crop(&self, image: &DynamicImage) -> DynamicImage {
        let (width, height) = image.dimensions();
        let crop_size = self.crop_size;
        let (left, top) = calculate_middle((width, height), (crop_size, crop_size));
        image.crop_imm(left, top, crop_size, crop_size)
    }

    pub fn to_tensor(&self, image: &DynamicImage) -> Result<Tensor> {
        let img = image.to_rgb8().into_raw();
        let (width, height) = image.dimensions();
        Tensor::from_vec(img, (height as usize, width as usize, 3), &Device::Cpu)?
            .to_dtype(DType::F32) // only for internal compute
    }

    pub fn rescale(&self, tensor: &Tensor) -> Result<Tensor> {
        let rescale_factor = self.rescale_factor as f64;
        tensor.affine(rescale_factor, 0.0)
    }

    pub fn normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        let image_mean = self.image_mean.clone();
        let image_std = self.image_std.clone();
        let mean = Tensor::from_vec(image_mean, (3,), &Device::Cpu)?;
        let std = Tensor::from_vec(image_std, (3,), &Device::Cpu)?;
        tensor.broadcast_sub(&mean)?.broadcast_div(&std)
    }

    pub fn to_channel_dimension_format(&self, tensor: &Tensor) -> Result<Tensor> {
        tensor.permute((2, 0, 1))
    }

    pub fn preprocess(&self, image: &DynamicImage) -> Result<Tensor> {
        let image = if self.do_resize {
            self.resize(image)
        } else {
            image.clone()
        };
        let image = if self.do_center_crop {
            self.center_crop(&image)
        } else {
            image
        };
        let tensor = self.to_tensor(&image)?;
        let tensor = if self.do_rescale {
            self.rescale(&tensor)?
        } else {
            tensor
        };
        let tensor = if self.do_normalize {
            self.normalize(&tensor)?
        } else {
            tensor
        };
        self.to_channel_dimension_format(&tensor)
    }
}

pub fn calculate_middle(image_size: (u32, u32), center_size: (u32, u32)) -> (u32, u32) {
    let (width, height) = image_size;
    let (center_width, center_height) = center_size;
    let left = if width <= center_width {
        0
    } else {
        ((width as f32 - center_width as f32) / 2.0).ceil() as u32
    };
    let top = if height <= center_height {
        0
    } else {
        ((height as f32 - center_height as f32) / 2.0).ceil() as u32
    };
    (left, top)
}

pub fn process_image(
    image: &DynamicImage,
    processor: &ImageProcessor,
    llava_config: &LLaVAConfig,
) -> candle::Result<Tensor> {
    if llava_config.image_aspect_ratio == *"square" {
        processor.preprocess(image)?.unsqueeze(0)
    } else if llava_config.image_aspect_ratio == *"anyres" {
        process_anyres_image(image, processor, &llava_config.image_grid_pinpoints)
    } else if llava_config.image_aspect_ratio == *"pad" {
        process_pad_image(image, processor)
    } else {
        bail!("Invalid image aspect ratio")
    }
}

fn process_pad_image(image: &DynamicImage, processor: &ImageProcessor) -> Result<Tensor> {
    let mean_color = processor
        .image_mean
        .iter()
        .map(|x| ((*x) * 255.0) as u8)
        .collect::<Vec<u8>>();
    let mean_color = Rgb::from([mean_color[0], mean_color[1], mean_color[2]]);
    let image_padded = expand2square(image, mean_color);
    processor.preprocess(&image_padded)
}

fn process_anyres_image(
    image: &DynamicImage,
    processor: &ImageProcessor,
    grid_pinpoints: &[(u32, u32)],
) -> Result<Tensor> {
    let original_size = image.dimensions();
    let best_resolution = select_best_resolution(original_size, grid_pinpoints);
    let image_padded = resize_and_pad_image(image, best_resolution);
    let image_original_resize = image.resize_exact(
        processor.size,
        processor.size,
        image::imageops::FilterType::CatmullRom,
    );
    let mut patches = vec![image_original_resize];
    for patch in divide_to_patches(&image_padded, processor.crop_size) {
        patches.push(patch);
    }
    let tensors = patches
        .iter()
        .map(|patch| processor.preprocess(patch))
        .collect::<Result<Vec<Tensor>>>()?;
    Tensor::stack(&tensors, 0)
}

fn expand2square(image: &DynamicImage, background_color: Rgb<u8>) -> DynamicImage {
    let (width, height) = image.dimensions();
    match width.cmp(&height) {
        std::cmp::Ordering::Less => {
            let mut new_image =
                DynamicImage::from(RgbImage::from_pixel(height, height, background_color));
            overlay(&mut new_image, image, ((height - width) / 2) as i64, 0);
            new_image
        }
        std::cmp::Ordering::Equal => image.clone(),
        std::cmp::Ordering::Greater => {
            let mut new_image =
                DynamicImage::from(RgbImage::from_pixel(width, width, background_color));
            overlay(&mut new_image, image, 0, ((width - height) / 2) as i64);
            new_image
        }
    }
}

fn resize_and_pad_image(image: &DynamicImage, target_resolution: (u32, u32)) -> DynamicImage {
    let (original_width, original_height) = image.dimensions();
    let original_width_f = original_width as f32;
    let original_height_f = original_height as f32;
    let (target_width, target_height) = target_resolution;
    let target_width_f = target_width as f32;
    let target_height_f = target_height as f32;
    let scale_w = target_width_f / original_width_f;
    let scale_h = target_height_f / original_height_f;
    let (new_width, new_height) = if scale_w < scale_h {
        (
            target_width,
            min((original_height_f * scale_w).ceil() as u32, target_height),
        )
    } else {
        (
            min((original_width_f * scale_h).ceil() as u32, target_width),
            target_height,
        )
    };
    let resized_image = image.resize_exact(
        new_width,
        new_height,
        image::imageops::FilterType::CatmullRom,
    );
    let mut new_image = DynamicImage::new_rgb8(target_width, target_height);
    let (paste_x, paste_y) =
        calculate_middle((target_width, target_height), (new_width, new_height));
    overlay(
        &mut new_image,
        &resized_image,
        paste_x.into(),
        paste_y.into(),
    );
    new_image
}

fn divide_to_patches(image: &DynamicImage, patch_size: u32) -> Vec<DynamicImage> {
    let (width, height) = image.dimensions();
    let mut patches = Vec::new();
    for y in (0..height).step_by(patch_size as usize) {
        for x in (0..width).step_by(patch_size as usize) {
            let patch = image.crop_imm(x, y, patch_size, patch_size);
            patches.push(patch);
        }
    }
    patches
}
