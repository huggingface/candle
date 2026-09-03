#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod model;
use model::{Multiples, YoloV26};

use candle::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use clap::{Parser, ValueEnum};
use image::DynamicImage;

const MODEL_INPUT_SIZE: u32 = 640;
const LETTERBOX_PAD_VALUE: f32 = 114.0 / 255.0;

#[derive(Clone, Copy, ValueEnum, Debug)]
enum Which {
    N,
    S,
    M,
    L,
    X,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Model weights, in safetensors format.
    #[arg(long)]
    model: Option<String>,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value_t = Which::N)]
    which: Which,

    images: Vec<String>,

    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.25)]
    confidence_threshold: f32,

    /// The size for the legend, 0 means no legend.
    #[arg(long, default_value_t = 14)]
    legend_size: u32,
}

impl Args {
    fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let path = match &self.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("developer0hye/candle-yolo-v26".to_string());
                let size = match self.which {
                    Which::N => "n",
                    Which::S => "s",
                    Which::M => "m",
                    Which::L => "l",
                    Which::X => "x",
                };
                api.get(&format!("yolo26{size}.safetensors"))?
            }
        };
        Ok(path)
    }
}

pub struct LetterboxInfo {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

/// Letterbox preprocess: resize maintaining aspect ratio, pad to 640×640.
fn letterbox_preprocess(img: &DynamicImage, device: &Device) -> Result<(Tensor, LetterboxInfo)> {
    let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
    let target = MODEL_INPUT_SIZE as f32;
    let scale = f32::min(target / orig_w, target / orig_h);
    let new_w = (orig_w * scale).round() as u32;
    let new_h = (orig_h * scale).round() as u32;
    let pad_x = (MODEL_INPUT_SIZE - new_w) as f32 / 2.0;
    let pad_y = (MODEL_INPUT_SIZE - new_h) as f32 / 2.0;
    let pad_x_int = pad_x.floor() as u32;
    let pad_y_int = pad_y.floor() as u32;

    let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
    let resized_rgb = resized.to_rgb8();

    let size = MODEL_INPUT_SIZE as usize;
    let channel_size = size * size;
    let mut chw = vec![LETTERBOX_PAD_VALUE; 3 * channel_size];

    for dy in 0..new_h {
        for dx in 0..new_w {
            let out_y = (dy + pad_y_int) as usize;
            let out_x = (dx + pad_x_int) as usize;
            if out_y < size && out_x < size {
                let pixel = resized_rgb.get_pixel(dx, dy);
                let pos = out_y * size + out_x;
                chw[pos] = pixel[0] as f32 / 255.0;
                chw[channel_size + pos] = pixel[1] as f32 / 255.0;
                chw[2 * channel_size + pos] = pixel[2] as f32 / 255.0;
            }
        }
    }

    let tensor = Tensor::from_vec(chw, (1, 3, size, size), device)?.to_dtype(DType::F32)?;
    let info = LetterboxInfo {
        scale,
        pad_x,
        pad_y,
    };
    Ok((tensor, info))
}

pub fn report_detect(
    pred: &Tensor,
    img: DynamicImage,
    letterbox: &LetterboxInfo,
    confidence_threshold: f32,
    legend_size: u32,
) -> Result<DynamicImage> {
    let pred = pred.to_device(&Device::Cpu)?;
    let (ndet, six) = pred.dims2()?;
    if six != 6 {
        candle::bail!("unexpected detection dim {six}, expected 6");
    }
    let data: Vec<f32> = pred.flatten_all()?.to_vec1()?;

    let (initial_w, initial_h) = (img.width() as f32, img.height() as f32);
    let mut img = img.to_rgb8();
    let font = Vec::from(include_bytes!("roboto-mono-stripped.ttf") as &[u8]);
    let font = ab_glyph::FontRef::try_from_slice(&font).map_err(candle::Error::wrap)?;

    for i in 0..ndet {
        let base = i * 6;
        let confidence = data[base + 4];
        if confidence < confidence_threshold {
            continue;
        }
        let class_id = data[base + 5] as usize;

        // Reverse letterbox: model coords → original image coords
        let x1 = ((data[base] - letterbox.pad_x) / letterbox.scale)
            .max(0.0)
            .min(initial_w);
        let y1 = ((data[base + 1] - letterbox.pad_y) / letterbox.scale)
            .max(0.0)
            .min(initial_h);
        let x2 = ((data[base + 2] - letterbox.pad_x) / letterbox.scale)
            .max(0.0)
            .min(initial_w);
        let y2 = ((data[base + 3] - letterbox.pad_y) / letterbox.scale)
            .max(0.0)
            .min(initial_h);

        let class_name = if class_id < candle_examples::coco_classes::NAMES.len() {
            candle_examples::coco_classes::NAMES[class_id]
        } else {
            "unknown"
        };

        println!("{class_name}: {confidence:.2} ({x1:.1}, {y1:.1}, {x2:.1}, {y2:.1})");

        let dx = x2 - x1;
        let dy = y2 - y1;
        if dx >= 0. && dy >= 0. {
            imageproc::drawing::draw_hollow_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(x1 as i32, y1 as i32).of_size(dx as u32, dy as u32),
                image::Rgb([255, 0, 0]),
            );
        }
        if legend_size > 0 {
            imageproc::drawing::draw_filled_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(x1 as i32, y1 as i32).of_size(dx as u32, legend_size),
                image::Rgb([170, 0, 0]),
            );
            let legend = format!("{class_name}   {:.0}%", 100. * confidence);
            imageproc::drawing::draw_text_mut(
                &mut img,
                image::Rgb([255, 255, 255]),
                x1 as i32,
                y1 as i32,
                ab_glyph::PxScale {
                    x: legend_size as f32 - 1.,
                    y: legend_size as f32 - 1.,
                },
                &font,
                &legend,
            );
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}

pub fn main() -> anyhow::Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let device = candle_examples::device(args.cpu)?;
    let multiples = match args.which {
        Which::N => Multiples::n(),
        Which::S => Multiples::s(),
        Which::M => Multiples::m(),
        Which::L => Multiples::l(),
        Which::X => Multiples::x(),
    };
    let model_path = args.model()?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
    let model = YoloV26::load(vb, multiples, 80)?;
    println!("model loaded");

    for image_name in args.images.iter() {
        println!("processing {image_name}");
        let mut image_path = std::path::PathBuf::from(image_name);
        let original_image = image::ImageReader::open(&image_path)?
            .decode()
            .map_err(candle::Error::wrap)?;

        let (image_t, letterbox) = letterbox_preprocess(&original_image, &device)?;
        let predictions = model.forward(&image_t)?.squeeze(0)?;
        println!("generated predictions {predictions:?}");

        let annotated = report_detect(
            &predictions,
            original_image,
            &letterbox,
            args.confidence_threshold,
            args.legend_size,
        )?;
        image_path.set_extension("pp.jpg");
        println!("writing {image_path:?}");
        annotated.save(image_path)?;
    }

    Ok(())
}
