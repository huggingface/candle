#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod coco_classes;
mod darknet;

use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use clap::Parser;
use image::{DynamicImage, ImageBuffer};

const CONFIDENCE_THRESHOLD: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.4;

#[derive(Debug, Clone, Copy)]
struct Bbox {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    confidence: f32,
}

// Intersection over union of two bounding boxes.
fn iou(b1: &Bbox, b2: &Bbox) -> f32 {
    let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
    let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
    let i_xmin = b1.xmin.max(b2.xmin);
    let i_xmax = b1.xmax.min(b2.xmax);
    let i_ymin = b1.ymin.max(b2.ymin);
    let i_ymax = b1.ymax.min(b2.ymax);
    let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
    i_area / (b1_area + b2_area - i_area)
}

// Assumes x1 <= x2 and y1 <= y2
pub fn draw_rect(
    img: &mut ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    x1: u32,
    x2: u32,
    y1: u32,
    y2: u32,
) {
    for x in x1..=x2 {
        let pixel = img.get_pixel_mut(x, y1);
        *pixel = image::Rgb([255, 0, 0]);
        let pixel = img.get_pixel_mut(x, y2);
        *pixel = image::Rgb([255, 0, 0]);
    }
    for y in y1..=y2 {
        let pixel = img.get_pixel_mut(x1, y);
        *pixel = image::Rgb([255, 0, 0]);
        let pixel = img.get_pixel_mut(x2, y);
        *pixel = image::Rgb([255, 0, 0]);
    }
}

pub fn report(pred: &Tensor, img: DynamicImage, w: usize, h: usize) -> Result<DynamicImage> {
    let (npreds, pred_size) = pred.dims2()?;
    let nclasses = pred_size - 5;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.get(index)?)?;
        let confidence = pred[4];
        if confidence > CONFIDENCE_THRESHOLD {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[5 + i] > pred[5 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 5] > 0. {
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                };
                bboxes[class_index].push(bbox)
            }
        }
    }
    // Perform non-maximum suppression.
    for bboxes_for_class in bboxes.iter_mut() {
        bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
        let mut current_index = 0;
        for index in 0..bboxes_for_class.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);
                if iou > NMS_THRESHOLD {
                    drop = true;
                    break;
                }
            }
            if !drop {
                bboxes_for_class.swap(current_index, index);
                current_index += 1;
            }
        }
        bboxes_for_class.truncate(current_index);
    }
    // Annotate the original image and print boxes information.
    let (initial_h, initial_w) = (img.height(), img.width());
    let w_ratio = initial_w as f32 / w as f32;
    let h_ratio = initial_h as f32 / h as f32;
    let mut img = img.to_rgb8();
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            println!("{}: {:?}", coco_classes::NAMES[class_index], b);
            let xmin = ((b.xmin * w_ratio) as u32).clamp(0, initial_w - 1);
            let ymin = ((b.ymin * h_ratio) as u32).clamp(0, initial_h - 1);
            let xmax = ((b.xmax * w_ratio) as u32).clamp(0, initial_w - 1);
            let ymax = ((b.ymax * h_ratio) as u32).clamp(0, initial_h - 1);
            draw_rect(&mut img, xmin, xmax, ymin, ymax);
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model weights, in safetensors format.
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    config: Option<String>,

    images: Vec<String>,
}

impl Args {
    fn config(&self) -> anyhow::Result<std::path::PathBuf> {
        let path = match &self.config {
            Some(config) => std::path::PathBuf::from(config),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("lmz/candle-yolo-v3".to_string());
                api.get("yolo-v3.cfg")?
            }
        };
        Ok(path)
    }

    fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let path = match &self.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("lmz/candle-yolo-v3".to_string());
                api.get("yolo-v3.safetensors")?
            }
        };
        Ok(path)
    }
}

pub fn main() -> Result<()> {
    let args = Args::parse();

    // Create the model and load the weights from the file.
    let model = args.model()?;
    let weights = unsafe { candle::safetensors::MmapedFile::new(model)? };
    let weights = weights.deserialize()?;
    let vb = VarBuilder::from_safetensors(vec![weights], DType::F32, &Device::Cpu);
    let config = args.config()?;
    let darknet = darknet::parse_config(config)?;
    let model = darknet.build_model(vb)?;

    for image_name in args.images.iter() {
        println!("processing {image_name}");
        let mut image_name = std::path::PathBuf::from(image_name);
        // Load the image file and resize it.
        let net_width = darknet.width()?;
        let net_height = darknet.height()?;

        let original_image = image::io::Reader::open(&image_name)?
            .decode()
            .map_err(candle::Error::wrap)?;
        let image = {
            let data = original_image
                .resize_exact(
                    net_width as u32,
                    net_height as u32,
                    image::imageops::FilterType::Triangle,
                )
                .to_rgb8()
                .into_raw();
            Tensor::from_vec(data, (net_width, net_height, 3), &Device::Cpu)?.permute((2, 0, 1))?
        };
        let image = (image.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
        let predictions = model.forward(&image)?.squeeze(0)?;
        println!("generated predictions {predictions:?}");
        let image = report(&predictions, original_image, net_width, net_height)?;
        image_name.set_extension("pp.jpg");
        println!("writing {image_name:?}");
        image.save(image_name)?
    }
    Ok(())
}
