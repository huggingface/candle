//! SAM 2: Segment Anything Model 2, single image prediction.
//! https://github.com/facebookresearch/sam2

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything_2::sam2::{BBox, Point, Sam2};
use candle_transformers::models::segment_anything_2::{Config, IMAGE_SIZE};
use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    Tiny,
    Small,
    BasePlus,
    Large,
}

impl Which {
    fn config(&self) -> Config {
        match self {
            Self::Tiny => Config::tiny(),
            Self::Small => Config::small(),
            Self::BasePlus => Config::base_plus(),
            Self::Large => Config::large(),
        }
    }

    fn repo_and_file(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("facebook/sam2.1-hiera-tiny", "sam2.1_hiera_tiny.pt"),
            Self::Small => ("facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt"),
            Self::BasePlus => (
                "facebook/sam2.1-hiera-base-plus",
                "sam2.1_hiera_base_plus.pt",
            ),
            Self::Large => ("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt"),
        }
    }
}

#[derive(Parser)]
struct Args {
    /// Path to a checkpoint, either the original `.pt` file or a safetensors file using the same
    /// tensor names. Downloaded from the hub when not set.
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long, value_enum, default_value_t = Which::Tiny)]
    which: Which,

    /// List of x,y coordinates, between 0 and 1 (0.5 is at the middle of the image). These points
    /// should be part of the generated mask.
    #[arg(long)]
    point: Vec<String>,

    /// List of x,y coordinates, between 0 and 1. These points should be part of the background.
    #[arg(long)]
    neg_point: Vec<String>,

    /// A box prompt as x0,y0,x1,y1 with coordinates between 0 and 1.
    #[arg(long)]
    bbox: Option<String>,

    /// Return the three multimask outputs rather than a single mask.
    #[arg(long)]
    multimask: bool,

    /// The detection threshold for the mask, 0 is the default value, negative values mean a larger
    /// mask, positive makes the mask more selective.
    #[arg(long, allow_hyphen_values = true, default_value_t = 0.)]
    threshold: f32,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,
}

fn parse_coords(s: &str, n: usize) -> anyhow::Result<Vec<f64>> {
    use std::str::FromStr;
    let values = s.split(',').collect::<Vec<_>>();
    if values.len() != n {
        anyhow::bail!("expected {n} comma separated values, got {s:?}")
    }
    values
        .iter()
        .map(|v| Ok(f64::from_str(v.trim())?))
        .collect()
}

/// SAM 2 stretches the image to a square, it does not preserve the aspect ratio the way SAM 1
/// does, so there is no padding to undo afterwards.
fn load_image(path: &str, device: &Device) -> anyhow::Result<(Tensor, usize, usize)> {
    let img = image::ImageReader::open(path)?
        .decode()
        .map_err(candle::Error::wrap)?;
    let (initial_h, initial_w) = (img.height() as usize, img.width() as usize);
    let img = img.resize_exact(
        IMAGE_SIZE as u32,
        IMAGE_SIZE as u32,
        image::imageops::FilterType::Triangle,
    );
    let data = img.to_rgb8().into_raw();
    let data = Tensor::from_vec(data, (IMAGE_SIZE, IMAGE_SIZE, 3), device)?.permute((2, 0, 1))?;
    Ok((data, initial_h, initial_w))
}

fn load_var_builder(
    path: &std::path::Path,
    device: &Device,
) -> anyhow::Result<VarBuilder<'static>> {
    match path.extension().and_then(|v| v.to_str()) {
        Some("safetensors") => {
            Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? })
        }
        // The original checkpoints store the state dict under a `model` key.
        _ => {
            let tensors = candle::pickle::read_all_with_key(path, Some("model"))?;
            let tensors = tensors
                .into_iter()
                .map(|(name, t)| Ok((name, t.to_device(device)?)))
                .collect::<candle::Result<std::collections::HashMap<_, _>>>()?;
            Ok(VarBuilder::from_tensors(tensors, DType::F32, device))
        }
    }
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
    let (image, initial_h, initial_w) = load_image(&args.image, &device)?;
    println!("loaded image {image:?}");

    let model = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => {
            let (repo, filename) = args.which.repo_and_file();
            candle_examples::hub::Api::new()?
                .model(repo)
                .get(filename)?
        }
    };
    let vb = load_var_builder(&model, &device)?;
    let sam = Sam2::new(&args.which.config(), vb)?;

    let mut points = vec![];
    for (list, is_foreground) in [(&args.point, true), (&args.neg_point, false)] {
        for p in list.iter() {
            let xy = parse_coords(p, 2)?;
            points.push(Point {
                x: xy[0],
                y: xy[1],
                is_foreground,
            })
        }
    }
    let bbox = match args.bbox.as_deref() {
        None => None,
        Some(b) => {
            let c = parse_coords(b, 4)?;
            Some(BBox {
                x0: c[0],
                y0: c[1],
                x1: c[2],
                y1: c[3],
            })
        }
    };
    if points.is_empty() && bbox.is_none() {
        anyhow::bail!("at least one --point, --neg-point or --bbox is required")
    }

    let start_time = std::time::Instant::now();
    let (masks, iou_predictions, object_score) =
        sam.forward(&image, &points, bbox, args.multimask, initial_h, initial_w)?;
    println!(
        "masks generated in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );
    println!("iou_predictions: {iou_predictions}");
    println!("object_score_logits: {object_score}");

    let mut img = image::ImageReader::open(&args.image)?
        .decode()
        .map_err(candle::Error::wrap)?;
    // The best mask is the first one when a single output was requested, and otherwise the one
    // with the highest predicted IoU.
    let best = iou_predictions.i(0)?.argmax(0)?.to_scalar::<u32>()? as usize;
    let mask = masks.i(best)?;
    let mask = (mask.ge(args.threshold)?.to_dtype(DType::U8)? * 255.)?;
    let (h, w) = mask.dims2()?;
    let mask_pixels = mask.flatten_all()?.to_vec1::<u8>()?;
    let mask_img: image::ImageBuffer<image::Luma<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(w as u32, h as u32, mask_pixels) {
            Some(image) => image,
            None => anyhow::bail!("error building the mask image"),
        };
    for x in 0..img.width() {
        for y in 0..img.height() {
            if mask_img.get_pixel(x, y).0[0] > 100 {
                let mut p = imageproc::drawing::Canvas::get_pixel(&img, x, y);
                p.0[2] = 255 - (255 - p.0[2]) / 2;
                p.0[1] /= 2;
                p.0[0] /= 2;
                imageproc::drawing::Canvas::draw_pixel(&mut img, x, y, p)
            }
        }
    }
    for p in points.iter() {
        let x = (p.x * img.width() as f64) as i32;
        let y = (p.y * img.height() as f64) as i32;
        let color = if p.is_foreground {
            image::Rgba([255, 0, 0, 200])
        } else {
            image::Rgba([0, 255, 0, 200])
        };
        imageproc::drawing::draw_filled_circle_mut(&mut img, (x, y), 3, color);
    }
    img.save("sam2_merged.jpg")?;
    println!("wrote sam2_merged.jpg");
    Ok(())
}
