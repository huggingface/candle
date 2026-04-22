//! SAM: Segment Anything Model
//! https://github.com/facebookresearch/segment-anything

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::DType;
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::sam;
use clap::Parser;

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

    /// List of x,y coordinates, between 0 and 1 (0.5 is at the middle of the image). These points
    /// should be part of the generated mask.
    #[arg(long)]
    point: Vec<String>,

    /// List of x,y coordinates, between 0 and 1 (0.5 is at the middle of the image). These points
    /// should not be part of the generated mask and should be part of the background instead.
    #[arg(long)]
    neg_point: Vec<String>,

    /// The detection threshold for the mask, 0 is the default value, negative values mean a larger
    /// mask, positive makes the mask more selective.
    #[arg(long, allow_hyphen_values = true, default_value_t = 0.)]
    threshold: f32,

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

    let (image, initial_h, initial_w) =
        candle_examples::load_image(&args.image, Some(sam::IMAGE_SIZE))?;
    let image = image.to_device(&device)?;
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
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
    let sam = if args.use_tiny {
        sam::Sam::new_tiny(vb)? // tiny vit_t
    } else {
        sam::Sam::new(768, 12, 12, &[2, 5, 8, 11], vb)? // sam_vit_b
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
            println!("{idx} {bbox:?}");
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
        let iter_points = args.point.iter().map(|p| (p, true));
        let iter_neg_points = args.neg_point.iter().map(|p| (p, false));
        let points = iter_points
            .chain(iter_neg_points)
            .map(|(point, b)| {
                use std::str::FromStr;
                let xy = point.split(',').collect::<Vec<_>>();
                if xy.len() != 2 {
                    anyhow::bail!("expected format for points is 0.4,0.2")
                }
                Ok((f64::from_str(xy[0])?, f64::from_str(xy[1])?, b))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let start_time = std::time::Instant::now();
        let (mask, iou_predictions) = sam.forward(&image, &points, false)?;
        println!(
            "mask generated in {:.2}s",
            start_time.elapsed().as_secs_f32()
        );
        println!("mask:\n{mask}");
        println!("iou_predictions: {iou_predictions}");

        let mask = (mask.ge(args.threshold)? * 255.)?;
        let (_one, h, w) = mask.dims3()?;
        let mask = mask.expand((3, h, w))?;

        let mut img = image::ImageReader::open(&args.image)?
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
        for (x, y, b) in points {
            let x = (x * img.width() as f64) as i32;
            let y = (y * img.height() as f64) as i32;
            let color = if b {
                image::Rgba([255, 0, 0, 200])
            } else {
                image::Rgba([0, 255, 0, 200])
            };
            imageproc::drawing::draw_filled_circle_mut(&mut img, (x, y), 3, color);
        }
        img.save("sam_merged.jpg")?
    }
    Ok(())
}
