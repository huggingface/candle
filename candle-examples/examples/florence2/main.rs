#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::florence2;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    image: String,

    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value = "microsoft/Florence-2-base")]
    model_id: String,
}

fn load_image<T: AsRef<std::path::Path>>(path: T, image_size: usize) -> anyhow::Result<Tensor> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
    Ok(img)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(args.model_id.clone());

    let config_path = repo.get("config.json")?;
    let config: florence2::Florence2Config =
        serde_json::from_reader(std::fs::File::open(config_path)?)?;

    println!(
        "Florence-2 config: vision_stages={}, text_d_model={}, projection_dim={}",
        config.vision_config.dim_embed.len(),
        config.text_config.d_model,
        config.projection_dim,
    );
    println!(
        "  DaViT: dim_embed={:?}, depths={:?}, window_size={}",
        config.vision_config.dim_embed,
        config.vision_config.depths,
        config.vision_config.window_size,
    );

    let model_path = match &args.model {
        Some(p) => std::path::PathBuf::from(p),
        None => repo.get("model.safetensors")?,
    };

    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };

    let model = florence2::Florence2VisionModel::new(&config, vb)?;

    // Image size: patch_stride product determines effective resolution.
    // Florence-2-base uses 768x768 images.
    let image_size = 768;
    let image = load_image(&args.image, image_size)?
        .unsqueeze(0)?
        .to_device(&device)?;

    println!("Encoding image ({image_size}x{image_size})...");
    let features = model.encode_image(&image)?;

    println!("Visual features shape: {:?}", features.shape());
    println!(
        "Feature norms: {:?}",
        features
            .sqr()?
            .sum(candle::D::Minus1)?
            .sqrt()?
            .flatten_all()?
            .to_vec1::<f32>()?
    );

    Ok(())
}
