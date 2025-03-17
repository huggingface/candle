#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;

use candle::{Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use itertools::Itertools;
use regex::Regex;
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone, Default)]
pub(super) struct PreProcessorConfig {
    do_convert_rgb: Option<bool>,
    do_normalize: Option<bool>,
    do_rescale: Option<bool>,
    do_resize: Option<bool>,
    #[serde(alias = "norm_mean")]
    image_mean: Option<[f64; 3]>,
    #[serde(alias = "norm_std")]
    image_std: Option<[f64; 3]>,
    rescale_factor: Option<f64>,
    #[serde(alias = "resample")]
    resampling: Option<usize>,
    size: Option<HashMap<String, u32>>,
    pan_and_scan_min_crop_size: Option<usize>,
    pan_and_scan_max_num_crops: Option<usize>,
    pan_and_scan_min_ratio_to_activate: Option<f64>,
    do_pan_and_scan: Option<bool>,
}

#[derive(Deserialize, Debug, Default)]
pub(super) struct ProcessorConfig {
    #[serde(alias = "image_seq_length")]
    image_seq_len: Option<usize>,
}

#[allow(dead_code)]
trait ToFilter {
    fn to_filter(self) -> Result<FilterType>;
}

impl ToFilter for Option<usize> {
    // https://github.com/python-pillow/Pillow/blob/4b68563e8a818fb9c528fa159ddf3f4eaefa35e6/src/PIL/Image.py#L164-L170
    // Default: https://github.com/huggingface/transformers/blob/0df888ffb72ea370555efdef45985378d3cc7b2b/src/transformers/models/idefics2/image_processing_idefics2.py#L226
    fn to_filter(self) -> Result<FilterType> {
        match self {
            Some(0) => Ok(FilterType::Nearest),
            Some(1) => Ok(FilterType::Lanczos3),
            Some(2) | None => Ok(FilterType::Triangle), // BiLinear
            Some(3) => Ok(FilterType::CatmullRom),      // BiCubic
            Some(4) => Ok(FilterType::Nearest),
            Some(x) => candle::bail!("Filter number {x} not supported"),
        }
    }
}

pub struct Gemma3ImageProcessor;

const IMAGE_TOKEN: &str = "<image_soft_token>";
const BOI_TOKEN: &str = "<start_of_image>";
const EOI_TOKEN: &str = "<end_of_image>";

pub struct Gemma3Processor {
    full_image_sequence: String,
}

impl Gemma3Processor {
    pub fn new(processor_config: ProcessorConfig) -> Self {
        let image_tokens_expanded =
            vec![IMAGE_TOKEN.to_string(); processor_config.image_seq_len.unwrap_or(256)].join("");
        let full_image_sequence = format!("\n\n{BOI_TOKEN}{image_tokens_expanded}{EOI_TOKEN}\n\n");

        Self {
            full_image_sequence,
        }
    }

    pub fn process_prompt(
        &self,
        mut prompt: String,
        pixel_values: &Tensor,
        num_crops: Vec<usize>,
    ) -> Result<String> {
        let re = Regex::new(BOI_TOKEN).map_err(candle::Error::wrap)?;
        let image_indexes: Vec<usize> = re.find_iter(&prompt).map(|mat| mat.start()).collect();

        assert_ne!(pixel_values.dim(0)?, image_indexes.len());

        for (num, idx) in num_crops.into_iter().zip(image_indexes).rev() {
            if num != 0 {
                let formatted_image_text = format!(
                        "Here is the original image {BOI_TOKEN} and here are some crops to help you see better {}", vec![BOI_TOKEN.to_string(); num].join(" ")
                    );
                prompt = format!(
                    "{}{formatted_image_text}{}",
                    &prompt[..idx],
                    &prompt[idx + BOI_TOKEN.len()..]
                );
            }
        }

        prompt = prompt.replace(BOI_TOKEN, &self.full_image_sequence);

        Ok(prompt)
    }
}

impl Gemma3ImageProcessor {
    fn pan_and_scan(
        &self,
        image: &DynamicImage,
        pan_and_scan_min_crop_size: usize,
        pan_and_scan_max_num_crops: usize,
        pan_and_scan_min_ratio_to_activate: f64,
    ) -> Vec<DynamicImage> {
        let (width, height) = image.dimensions();

        let (num_crops_w, num_crops_h) = if width >= height {
            if (width as f64 / height as f64) < pan_and_scan_min_ratio_to_activate {
                return vec![];
            }

            // Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
            let mut num_crops_w = (width as f64 / height as f64 + 0.5).floor() as usize;
            num_crops_w = num_crops_w
                .min((width as f64 / pan_and_scan_min_crop_size as f64).floor() as usize);

            // Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
            num_crops_w = num_crops_w.max(2);
            num_crops_w = num_crops_w.min(pan_and_scan_max_num_crops);

            (num_crops_w, 1)
        } else {
            if (width as f64 / height as f64) < pan_and_scan_min_ratio_to_activate {
                return vec![];
            }

            // Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
            let mut num_crops_h = (height as f64 / width as f64 + 0.5).floor() as usize;
            num_crops_h = num_crops_h
                .min((height as f64 / pan_and_scan_min_crop_size as f64).floor() as usize);

            // Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
            num_crops_h = num_crops_h.max(2);
            num_crops_h = num_crops_h.min(pan_and_scan_max_num_crops);

            (1, num_crops_h)
        };

        let crop_size_w = (width as f64 / num_crops_w as f64).ceil() as usize;
        let crop_size_h = (height as f64 / num_crops_h as f64).ceil() as usize;

        if crop_size_w.min(crop_size_h) < pan_and_scan_min_crop_size {
            return vec![];
        }

        let crop_positions_w = (0..num_crops_w)
            .map(|i| i * crop_size_w)
            .collect::<Vec<_>>();
        let crop_positions_h = (0..num_crops_h)
            .map(|i| i * crop_size_h)
            .collect::<Vec<_>>();

        let mut image_crops = Vec::new();
        for (pos_h, pos_w) in crop_positions_h
            .into_iter()
            .cartesian_product(crop_positions_w)
        {
            image_crops.push(image.crop_imm(
                pos_w as u32,
                pos_h as u32,
                crop_size_w as u32,
                crop_size_h as u32,
            ));
        }

        image_crops
    }

    fn process_images_for_pan_and_scan(
        &self,
        images: Vec<DynamicImage>,
        pan_and_scan_min_crop_size: usize,
        pan_and_scan_max_num_crops: usize,
        pan_and_scan_min_ratio_to_activate: f64,
    ) -> (Vec<DynamicImage>, Vec<usize>) {
        let mut pas_images_list = Vec::new();
        let mut num_crops = Vec::new();

        for image in images {
            let pas_images = self.pan_and_scan(
                &image,
                pan_and_scan_min_crop_size,
                pan_and_scan_max_num_crops,
                pan_and_scan_min_ratio_to_activate,
            );
            num_crops.push(pas_images.len());
            pas_images_list.extend([vec![image], pas_images].concat());
        }

        (pas_images_list, num_crops)
    }
}

impl Gemma3ImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    const DEFAULT_STD: [f64; 3] = [0.5, 0.5, 0.5];

    pub fn preprocess_image(
        &self,
        image: DynamicImage,
        config: &PreProcessorConfig,
        device: &Device,
    ) -> Result<(Tensor, Vec<usize>)> {
        // Just wrap it to a vec for easier internal usage
        let mut images = vec![image];

        let do_resize = config.do_resize.unwrap();
        let size = config.size.as_ref().unwrap();
        let (height, width) = (size["height"], size["width"]);
        let resample = config.resampling.to_filter()?;
        let do_rescale = config.do_rescale.unwrap();
        let rescale_factor = config.rescale_factor.unwrap();
        let do_normalize = config.do_normalize.unwrap();
        let image_mean = config.image_mean.unwrap_or(Self::DEFAULT_MEAN);
        let image_std = config.image_std.unwrap_or(Self::DEFAULT_STD);
        let do_convert_rgb = config.do_convert_rgb.unwrap_or(true);
        let do_pan_and_scan = config.do_pan_and_scan.unwrap_or(do_convert_rgb);
        // https://github.com/huggingface/transformers/blob/ea219ed164bead55a5513e8cfaa17a25d5613b9e/src/transformers/models/gemma3/processing_gemma3.py#L42
        let pan_and_scan_min_crop_size = config.pan_and_scan_min_crop_size.unwrap_or(256);
        let pan_and_scan_max_num_crops = config.pan_and_scan_max_num_crops.unwrap_or(4);
        let pan_and_scan_min_ratio_to_activate =
            config.pan_and_scan_min_ratio_to_activate.unwrap_or(1.2);

        for image in images.iter_mut() {
            // Convert to rgb
            if config.do_convert_rgb.is_some_and(|x| x) {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }
        }

        let num_crops = if do_pan_and_scan {
            let (new_images, num_crops) = self.process_images_for_pan_and_scan(
                images,
                pan_and_scan_min_crop_size,
                pan_and_scan_max_num_crops,
                pan_and_scan_min_ratio_to_activate,
            );
            images = new_images;
            num_crops
        } else {
            vec![0]
        };

        let mean = Tensor::new(image_mean.to_vec(), device)?.reshape((3, 1, 1))?;
        let std = Tensor::new(image_std.to_vec(), device)?.reshape((3, 1, 1))?;

        let mut pixel_values = Vec::new();
        for mut image in images {
            if do_resize {
                image = image.resize_exact(width, height, resample);
            }

            let img = image.to_rgb8();
            let data = img.into_raw();
            let mut image = Tensor::from_vec(data, (height as usize, width as usize, 3), device)?
                .permute((2, 0, 1))?;

            if do_rescale {
                image = (image * rescale_factor)?;
            }

            if do_normalize {
                image = image.broadcast_sub(&mean)?.broadcast_div(&std)?;
            }

            pixel_values.push(image.unsqueeze(0)?);
        }

        Ok((Tensor::cat(&pixel_values, 0)?, num_crops))
    }
}
