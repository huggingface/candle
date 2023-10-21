#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;

use candle::DType;
use candle_nn::VarBuilder;
use candle_transformers::models::blip;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;

    let image = candle_examples::imagenet::load_image224(args.image)?;
    println!("loaded image {image:?}");

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.repo(hf_hub::Repo::with_revision(
                "Salesforce/blip-image-captioning-large".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/18".to_string(),
            ));
            api.get("model.safetensors")?
        }
        Some(model) => model.into(),
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let config = blip::Config::image_captioning_large();
    let model = blip::BlipForConditionalGeneration::new(&config, vb)?;
    println!("model built");
    // TODO: Maybe add support for the conditional prompt.
    let out = model.generate(&image.unsqueeze(0)?, None, None)?;
    println!(">>>\n{out}");
    Ok(())
}
