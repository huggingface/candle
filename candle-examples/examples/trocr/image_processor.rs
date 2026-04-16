use image::{DynamicImage, ImageBuffer};
use serde::Deserialize;
use std::collections::HashMap;

use candle::{DType, Device, Result, Tensor};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ProcessorConfig {
    do_resize: bool,
    height: u32,
    width: u32,
    do_rescale: bool,
    do_normalize: bool,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            do_resize: true,
            height: 384,
            width: 384,
            do_rescale: true,
            do_normalize: true,
            image_mean: vec![0.5, 0.5, 0.5],
            image_std: vec![0.5, 0.5, 0.5],
        }
    }
}

pub struct ViTImageProcessor {
    do_resize: bool,
    height: u32,
    width: u32,
    do_normalize: bool,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
}

impl ViTImageProcessor {
    pub fn new(config: &ProcessorConfig) -> Self {
        Self {
            do_resize: config.do_resize,
            height: config.height,
            width: config.width,
            do_normalize: config.do_normalize,
            image_mean: config.image_mean.clone(),
            image_std: config.image_std.clone(),
        }
    }

    pub fn preprocess(&self, images: Vec<&str>) -> Result<Tensor> {
        let height = self.height as usize;
        let width = self.width as usize;
        let channels = 3;

        let images = self.load_images(images)?;

        let resized_images: Vec<DynamicImage> = if self.do_resize {
            images
                .iter()
                .map(|image| self.resize(image.clone(), None).unwrap())
                .collect()
        } else {
            images
        };

        let normalized_images: Vec<Tensor> = if self.do_normalize {
            resized_images
                .iter()
                .map(|image| self.normalize(image.clone(), None, None).unwrap())
                .collect()
        } else {
            let resized_images: Vec<ImageBuffer<image::Rgb<u8>, Vec<u8>>> =
                resized_images.iter().map(|image| image.to_rgb8()).collect();
            let data = resized_images
                .into_iter()
                .map(|image| image.into_raw())
                .collect::<Vec<Vec<u8>>>();

            data.iter()
                .map(|image| {
                    Tensor::from_vec(image.clone(), (height, width, channels), &Device::Cpu)
                        .unwrap()
                        .permute((2, 0, 1))
                        .unwrap()
                })
                .collect::<Vec<Tensor>>()
        };

        Tensor::stack(&normalized_images, 0)
    }

    fn resize(
        &self,
        image: image::DynamicImage,
        size: Option<HashMap<String, u32>>,
    ) -> Result<image::DynamicImage> {
        let (height, width) = match &size {
            Some(size) => (size.get("height").unwrap(), size.get("width").unwrap()),
            None => (&self.height, &self.width),
        };

        let resized_image =
            image.resize_exact(*width, *height, image::imageops::FilterType::Triangle);

        Ok(resized_image)
    }

    fn normalize(
        &self,
        image: image::DynamicImage,
        mean: Option<Vec<f32>>,
        std: Option<Vec<f32>>,
    ) -> Result<Tensor> {
        let mean = match mean {
            Some(mean) => mean,
            None => self.image_mean.clone(),
        };

        let std = match std {
            Some(std) => std,
            None => self.image_std.clone(),
        };

        let mean = Tensor::from_vec(mean, (3, 1, 1), &Device::Cpu)?;
        let std = Tensor::from_vec(std, (3, 1, 1), &Device::Cpu)?;

        let image = image.to_rgb8();
        let data = image.into_raw();

        let height = self.height as usize;
        let width = self.width as usize;
        let channels = 3;

        let data =
            Tensor::from_vec(data, &[height, width, channels], &Device::Cpu)?.permute((2, 0, 1))?;

        (data.to_dtype(DType::F32)? / 255.)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)
    }

    pub fn load_images(&self, image_path: Vec<&str>) -> Result<Vec<image::DynamicImage>> {
        let mut images: Vec<image::DynamicImage> = Vec::new();
        for path in image_path {
            let img = image::ImageReader::open(path)?.decode().unwrap();
            images.push(img);
        }

        Ok(images)
    }
}
