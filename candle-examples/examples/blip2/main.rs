#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;

use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::blip2;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    image: String,

    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value = "Salesforce/blip2-opt-2.7b")]
    model_id: String,
}

fn load_image<T: AsRef<std::path::Path>>(
    path: T,
    image_size: usize,
) -> anyhow::Result<Tensor> {
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
    let config: blip2::Blip2Config = serde_json::from_reader(std::fs::File::open(config_path)?)?;

    println!(
        "BLIP-2 config: vision_hidden_size={}, qformer_hidden_size={}, num_query_tokens={}",
        config.vision_config.hidden_size,
        config.qformer_config.hidden_size,
        config.num_query_tokens,
    );

    let model_path = match &args.model {
        Some(p) => std::path::PathBuf::from(p),
        None => repo.get("model.safetensors")?,
    };

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)?
    };

    let model = blip2::Blip2Model::new(&config, vb)?;

    let image_size = config.vision_config.image_size;
    let image = load_image(&args.image, image_size)?
        .unsqueeze(0)?
        .to_device(&device)?;

    println!("Extracting Q-Former features from image...");
    let qformer_features = model.get_qformer_features(&image)?;

    println!(
        "Q-Former output shape: {:?}",
        qformer_features.shape()
    );

    let language_inputs = model.get_language_model_inputs(&image)?;
    println!(
        "Language projection output shape: {:?}",
        language_inputs.shape()
    );

    // Print feature norms for verification
    let norms = qformer_features
        .sqr()?
        .sum(2)?
        .sqrt()?;
    println!("Feature norms (first 5 query tokens): {:?}", norms.i((0, ..5))?.to_vec1::<f32>()?);

    Ok(())
}
