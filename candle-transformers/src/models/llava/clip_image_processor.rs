use candle::{DType, Device, Result, Tensor};
use hf_hub::api::sync::Api;
use image::{DynamicImage, GenericImageView};
use serde::{Deserialize, Serialize};

//This struct is mainly for LLaVA aplications, hence it's not completely compatible with python transformer CLIPImageProcessor  few several preprocess that LLaVA used, including "openai/clip-vit-large-patch14-336" and "openai/clip-vit-large-patch14".

#[derive(Serialize, Deserialize, Debug)]
pub struct CLIPImageProcessor {
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

impl CLIPImageProcessor {
    pub fn from_pretrained(clip_id: &str) -> Result<Self> {
        let api = Api::new().map_err(|e| candle::Error::Msg(e.to_string()))?;
        let api = api.model(clip_id.to_string());
        let config_filename = api
            .get("preprocessor_config.json")
            .map_err(|e| candle::Error::Msg(e.to_string()))?;
        let image_processor = serde_json::from_slice(
            &std::fs::read(config_filename).map_err(|e| candle::Error::Io(e))?,
        )
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
        Ok(image_processor)
    }

    ///shortest edge to self.resize, other edge is resized to maintain aspect ratio
    pub fn resize(&self, image: &DynamicImage) -> DynamicImage {
        let (width, height) = image.dimensions();
        let size = self.size;
        if width == size && height == size {
            image.clone()
        } else {
            let (new_width, new_height) = if width < height {
                let _new_height = (((size * height) as f32) / width as f32).ceil() as u32;
                (size, _new_height)
            } else {
                let _new_width = (((size * width) as f32) / height as f32).ceil() as u32;
                (_new_width, size)
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::io::Reader as ImageReader;
    use std::path::Path;
    const CLIP_ID: &str = "openai/clip-vit-large-patch14-336";

    #[test]
    fn test_resize() {
        let image_path = Path::new("images/Rectangle-1.png");
        let image = ImageReader::open(image_path).unwrap().decode().unwrap();
        let clip_image_processor = CLIPImageProcessor::from_pretrained(CLIP_ID).unwrap();
        let resized_image = clip_image_processor.resize(&image);
        resized_image.save("tmp/Rectangle-1-resized.png").unwrap();
    }
    #[test]
    fn test_center_crop() {
        let image_path = Path::new("images/llava_v1_5_radar.jpg");
        let image = ImageReader::open(image_path).unwrap().decode().unwrap();
        let clip_image_processor = CLIPImageProcessor::from_pretrained(CLIP_ID).unwrap();
        let image_cropped = clip_image_processor.center_crop(&image);
        image_cropped
            .save("tmp/llava_v1_5_radar_cropped.jpg")
            .unwrap();
    }
    #[test]
    fn test_preprocess() {
        let image_path = Path::new("images/llava_v1_5_radar.jpg");
        let image = ImageReader::open(image_path).unwrap().decode().unwrap();
        let clip_image_processor = CLIPImageProcessor::from_pretrained(CLIP_ID).unwrap();
        let tensor = clip_image_processor.preprocess(&image).unwrap();
        println!("{:?}", tensor.shape());
    }
}
