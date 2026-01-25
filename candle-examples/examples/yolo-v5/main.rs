#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod model;

use std::path::PathBuf;

use candle::{DType, Tensor};
use candle_examples::device;
use candle_nn::VarBuilder;
use candle_transformers::object_detection::{non_maximum_suppression, Bbox};
use clap::{Parser, ValueEnum};
use image::GenericImageView;

#[derive(ValueEnum, Clone)]
enum Which {
    N,
    S,
    M,
    L,
}

impl std::fmt::Display for Which {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Which::N => "n",
            Which::S => "s",
            Which::M => "m",
            Which::L => "l",
        };
        write!(f, "{s}")
    }
}

#[derive(Parser)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Path to model weights file.
    #[arg(long)]
    model: Option<String>,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value_t = Which::S)]
    which: Which,

    /// Path to input image.
    #[arg(long)]
    image: PathBuf,

    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.25)]
    confidence_threshold: f32,

    /// Threshold for non-maximum suppression.
    #[arg(long, default_value_t = 0.45)]
    nms_threshold: f32,
}

impl Args {
    fn weights(&self) -> anyhow::Result<std::path::PathBuf> {
        let path = match &self.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("mayocream/yolo-v5".to_string());
                let size = self.which.to_string();
                api.get(&format!("yolov5{size}.safetensors"))?
            }
        };
        Ok(path)
    }
}

fn preprocess(
    image: &image::DynamicImage,
    device: &candle::Device,
) -> candle::Result<candle::Tensor> {
    // Convert to RGB and keep the raw bytes in HWC layout, then permute to NCHW.
    let image = image.resize_exact(640, 640, image::imageops::FilterType::Nearest);
    let rgb = image.to_rgb8();
    Tensor::from_vec(rgb.into_raw(), (1, 640, 640, 3), device)?
        .to_dtype(DType::F32)?
        .permute((0, 3, 1, 2))?
        * (1. / 255.)
}

fn postprocess(
    image: &image::DynamicImage,
    predictions: Tensor,
    conf_threshold: f32,
    nms_threshold: f32,
) -> candle::Result<Vec<Bbox<usize>>> {
    let predictions = predictions.squeeze(0)?;
    let (_, num_outputs) = predictions.dims2()?;
    // YOLOv5 format: [cx, cy, w, h, objectness, class1_score, class2_score, ...]
    let num_classes = num_outputs - 5;

    let (orig_w, orig_h) = image.dimensions();
    let w_ratio = orig_w as f32 / 640.;
    let h_ratio = orig_h as f32 / 640.;

    let mut bboxes: Vec<Vec<Bbox<usize>>> = (0..num_classes).map(|_| Vec::new()).collect();
    let predictions: Vec<Vec<f32>> = predictions.to_vec2()?;
    for pred in predictions {
        let (class_index, confidence) = {
            let (cls_idx, cls_score) = pred[5..]
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap_or((0, 0.0));
            (cls_idx, pred[4] * cls_score)
        };
        if confidence < conf_threshold {
            continue;
        }

        let xmin = ((pred[0] - pred[2] / 2.) * w_ratio).clamp(0., orig_w as f32);
        let xmax = ((pred[0] + pred[2] / 2.) * w_ratio).clamp(0., orig_w as f32);
        let ymin = ((pred[1] - pred[3] / 2.) * h_ratio).clamp(0., orig_h as f32);
        let ymax = ((pred[1] + pred[3] / 2.) * h_ratio).clamp(0., orig_h as f32);

        let bbox = Bbox {
            xmin,
            xmax,
            ymin,
            ymax,
            confidence,
            data: class_index,
        };
        bboxes[class_index].push(bbox);
    }

    non_maximum_suppression(&mut bboxes, nms_threshold);

    Ok(bboxes.into_iter().flatten().collect())
}

fn report(image: &mut image::DynamicImage, bboxes: &[Bbox<usize>]) {
    for bbox in bboxes {
        imageproc::drawing::draw_hollow_rect_mut(
            image,
            imageproc::rect::Rect::at(bbox.xmin as i32, bbox.ymin as i32).of_size(
                (bbox.xmax - bbox.xmin) as u32,
                (bbox.ymax - bbox.ymin) as u32,
            ),
            image::Rgba([255, 0, 0, 255]),
        );
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = device(args.cpu)?;

    let weights = args.weights()?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
    let model = model::YoloV5::load(vb, 80, 3)?;
    let mut image = image::open(&args.image)?;
    let image_t = preprocess(&image, &device)?;
    let (predictions, _) = model.forward(&image_t)?;
    let bboxes = postprocess(
        &image,
        predictions,
        args.confidence_threshold,
        args.nms_threshold,
    )?;

    report(&mut image, &bboxes);
    image.save(&args.image.with_extension("pp.jpg"))?;

    Ok(())
}
