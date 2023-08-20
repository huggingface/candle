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

const CONFIG_NAME: &str = "candle-examples/examples/yolo/yolo-v3.cfg";
const CONFIDENCE_THRESHOLD: f64 = 0.5;
const NMS_THRESHOLD: f64 = 0.4;

#[derive(Debug, Clone, Copy)]
struct Bbox {
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
    confidence: f64,
}

// Intersection over union of two bounding boxes.
fn iou(b1: &Bbox, b2: &Bbox) -> f64 {
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
pub fn draw_rect(_: &mut Tensor, _x1: usize, _x2: usize, _y1: usize, _y2: usize) {
    todo!()
}

pub fn report(pred: &Tensor, img: &Tensor, w: usize, h: usize) -> Result<Tensor> {
    let (npreds, pred_size) = pred.dims2()?;
    let nclasses = pred_size - 5;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f64>::try_from(pred.get(index)?)?;
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
    let (_, initial_h, initial_w) = img.dims3()?;
    let mut img = (img.to_dtype(DType::F32)? * (1. / 255.))?;
    let w_ratio = initial_w as f64 / w as f64;
    let h_ratio = initial_h as f64 / h as f64;
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            println!("{}: {:?}", coco_classes::NAMES[class_index], b);
            let xmin = ((b.xmin * w_ratio) as usize).clamp(0, initial_w - 1);
            let ymin = ((b.ymin * h_ratio) as usize).clamp(0, initial_h - 1);
            let xmax = ((b.xmax * w_ratio) as usize).clamp(0, initial_w - 1);
            let ymax = ((b.ymax * h_ratio) as usize).clamp(0, initial_h - 1);
            draw_rect(&mut img, xmin, xmax, ymin, ymax.min(ymin + 2));
            draw_rect(&mut img, xmin, xmax, ymin.max(ymax - 2), ymax);
            draw_rect(&mut img, xmin, xmax.min(xmin + 2), ymin, ymax);
            draw_rect(&mut img, xmin.max(xmax - 2), xmax, ymin, ymax);
        }
    }
    Ok((img * 255.)?.to_dtype(DType::U8)?)
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model weights, in safetensors format.
    #[arg(long)]
    model: String,

    images: Vec<String>,
}

pub fn main() -> Result<()> {
    let args = Args::parse();

    // Create the model and load the weights from the file.
    let weights = unsafe { candle::safetensors::MmapedFile::new(&args.model)? };
    let weights = weights.deserialize()?;
    let vb = VarBuilder::from_safetensors(vec![weights], DType::F32, &Device::Cpu);
    let darknet = darknet::parse_config(CONFIG_NAME)?;
    let model = darknet.build_model(vb)?;

    for image in args.images.iter() {
        // Load the image file and resize it.
        let net_width = darknet.width()?;
        let net_height = darknet.height()?;
        let image = candle_examples::load_image_and_resize(image, net_width, net_height)?;
        let image = (image.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
        let predictions = model.forward(&image)?.squeeze(0)?;
        let _image = report(&predictions, &image, net_width, net_height)?;
        println!("converted {image}");
    }
    Ok(())
}
