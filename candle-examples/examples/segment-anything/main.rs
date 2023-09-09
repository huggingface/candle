//! SAM: Segment Anything Model
//! https://github.com/facebookresearch/segment-anything

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub mod model_image_encoder;
pub mod model_mask_decoder;
pub mod model_prompt_encoder;
pub mod model_sam;
pub mod model_tiny_vit;
pub mod model_transformer;

use candle::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};
use clap::Parser;

pub fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    let inner = if bias {
        candle_nn::linear(in_dim, out_dim, vb)?
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)?
    };
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

#[derive(Debug)]
pub struct LayerNorm2d {
    weight: Tensor,
    bias: Tensor,
    num_channels: usize,
    eps: f64,
}

impl LayerNorm2d {
    pub fn new(num_channels: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(num_channels, "weight")?;
        let bias = vb.get(num_channels, "bias")?;
        Ok(Self {
            weight,
            bias,
            num_channels,
            eps,
        })
    }
}

impl Module for LayerNorm2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let u = xs.mean_keepdim(1)?;
        let xs = xs.broadcast_sub(&u)?;
        let s = xs.sqr()?.mean_keepdim(1)?;
        let xs = xs.broadcast_div(&(s + self.eps)?.sqrt()?)?;
        xs.broadcast_mul(&self.weight.reshape((1, self.num_channels, 1, 1))?)?
            .broadcast_add(&self.bias.reshape((1, self.num_channels, 1, 1))?)
    }
}

#[derive(Debug)]
pub struct MlpBlock {
    lin1: Linear,
    lin2: Linear,
    activation: candle_nn::Activation,
    span: tracing::Span,
}

impl MlpBlock {
    pub fn new(
        embedding_dim: usize,
        mlp_dim: usize,
        activation: candle_nn::Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let lin1 = linear(vb.pp("lin1"), embedding_dim, mlp_dim, true)?;
        let lin2 = linear(vb.pp("lin2"), mlp_dim, embedding_dim, true)?;
        let span = tracing::span!(tracing::Level::TRACE, "mlp-block");
        Ok(Self {
            lin1,
            lin2,
            activation,
            span,
        })
    }
}

impl Module for MlpBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        xs.apply(&self.lin1)?
            .apply(&self.activation)?
            .apply(&self.lin2)
    }
}

#[derive(Debug)]
pub struct Linear {
    inner: candle_nn::Linear,
    span: tracing::Span,
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    generate_masks: bool,

    #[arg(long, default_value_t = 0.5)]
    point_x: f64,

    #[arg(long, default_value_t = 0.5)]
    point_y: f64,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Use the TinyViT based models from MobileSAM
    #[arg(long)]
    use_tiny: bool,
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

    let (image, initial_h, initial_w) = if args.image.ends_with(".safetensors") {
        let mut tensors = candle::safetensors::load(&args.image, &device)?;
        let image = match tensors.remove("image") {
            Some(image) => image,
            None => {
                if tensors.len() != 1 {
                    anyhow::bail!("multiple tensors in '{}'", args.image)
                }
                tensors.into_values().next().unwrap()
            }
        };
        let image = if image.rank() == 4 {
            image.get(0)?
        } else {
            image
        };
        let (_c, h, w) = image.dims3()?;
        (image, h, w)
    } else {
        let (image, h, w) = candle_examples::load_image(&args.image, Some(model_sam::IMAGE_SIZE))?;
        (image.to_device(&device)?, h, w)
    };
    println!("loaded image {image:?}");

    let model = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/candle-sam".to_string());
            let filename = if args.use_tiny {
                "mobile_sam-tiny-vitt.safetensors"
            } else {
                "sam_vit_b_01ec64.safetensors"
            };
            api.get(filename)?
        }
    };
    let weights = unsafe { candle::safetensors::MmapedFile::new(model)? };
    let weights = weights.deserialize()?;
    let vb = VarBuilder::from_safetensors(vec![weights], DType::F32, &device);
    let sam = if args.use_tiny {
        model_sam::Sam::new_tiny(vb)? // tiny vit_t
    } else {
        model_sam::Sam::new(768, 12, 12, &[2, 5, 8, 11], vb)? // sam_vit_b
    };

    if args.generate_masks {
        // Default options similar to the Python version.
        let bboxes = sam.generate_masks(
            &image,
            /* points_per_side */ 32,
            /* crop_n_layer */ 0,
            /* crop_overlap_ratio */ 512. / 1500.,
            /* crop_n_points_downscale_factor */ 1,
        )?;
        for (idx, bbox) in bboxes.iter().enumerate() {
            println!("{bbox:?}");
            let mask = (&bbox.data.to_dtype(DType::U8)? * 255.)?;
            let (h, w) = mask.dims2()?;
            let mask = mask.broadcast_as((3, h, w))?;
            candle_examples::save_image_resize(
                &mask,
                format!("sam_mask{idx}.png"),
                initial_h,
                initial_w,
            )?;
        }
    } else {
        let point = Some((args.point_x, args.point_y));
        let start_time = std::time::Instant::now();
        let (mask, iou_predictions) = sam.forward(&image, point, false)?;
        println!(
            "mask generated in {:.2}s",
            start_time.elapsed().as_secs_f32()
        );
        println!("mask:\n{mask}");
        println!("iou_predictions: {iou_predictions:?}");

        // Save the mask as an image.
        let mask = (mask.ge(0f32)? * 255.)?;
        let (_one, h, w) = mask.dims3()?;
        let mask = mask.expand((3, h, w))?;
        candle_examples::save_image_resize(&mask, "sam_mask.png", initial_h, initial_w)?;

        if !args.image.ends_with(".safetensors") {
            let mut img = image::io::Reader::open(&args.image)?
                .decode()
                .map_err(candle::Error::wrap)?;
            let mask_pixels = mask.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
            let mask_img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
                match image::ImageBuffer::from_raw(w as u32, h as u32, mask_pixels) {
                    Some(image) => image,
                    None => anyhow::bail!("error saving merged image"),
                };
            let mask_img = image::DynamicImage::from(mask_img).resize_to_fill(
                img.width(),
                img.height(),
                image::imageops::FilterType::CatmullRom,
            );
            for x in 0..img.width() {
                for y in 0..img.height() {
                    let mask_p = imageproc::drawing::Canvas::get_pixel(&mask_img, x, y);
                    if mask_p.0[0] > 100 {
                        let mut img_p = imageproc::drawing::Canvas::get_pixel(&img, x, y);
                        img_p.0[2] = 255 - (255 - img_p.0[2]) / 2;
                        img_p.0[1] /= 2;
                        img_p.0[0] /= 2;
                        imageproc::drawing::Canvas::draw_pixel(&mut img, x, y, img_p)
                    }
                }
            }
            match point {
                Some((x, y)) => {
                    let (x, y) = (
                        (x * img.width() as f64) as i32,
                        (y * img.height() as f64) as i32,
                    );
                    imageproc::drawing::draw_filled_circle(
                        &img,
                        (x, y),
                        3,
                        image::Rgba([255, 0, 0, 200]),
                    )
                    .save("sam_merged.jpg")?
                }
                None => img.save("sam_merged.jpg")?,
            };
        }
    }
    Ok(())
}
