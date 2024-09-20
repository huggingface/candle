use anyhow::Ok;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;

use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::chinese_clip::{text_model, vision_model};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long, use_value_delimiter = true)]
    images: Option<Vec<String>>,

    #[arg(long)]
    cpu: bool,

    #[arg(long, use_value_delimiter = true)]
    sequences: Option<Vec<String>>,
}

fn main() -> anyhow::Result<()> {
    println!("Hello, world!");
    let args = Args::parse();

    tracing_subscriber::fmt::init();

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let repo = hf_hub::Repo::with_revision(
                "OFA-Sys/chinese-clip-vit-base-patch16".to_string(),
                hf_hub::RepoType::Model,
                "main".to_string(),
            );
            let api = api.repo(repo);
            api.get("pytorch_model.bin")?
        }
        Some(model) => model.into(),
    };

    // let model_file =
    //     "/home/shawn/workspace/rum-backend-python/tmp/chinese-clip-vit-base-patch16.safetensors";
    println!("Model file: {:?}", model_file);

    let device = Device::Cpu;
    let var = VarBuilder::from_pth(model_file, DType::F32, &device)?;
    // let var = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device) }?;
    println!(
        "contains embeddinng: {}",
        var.contains_tensor("visual_projection.weight")
    );

    let vision_transformer = vision_model::ChineseClipVisionTransformer::new(
        var.pp("vision_model"),
        &vision_model::ChineseClipVisionConfig::clip_vit_base_patch16(),
    )?;
    println!("vision_transformer: {:?}", vision_transformer);

    let text_transformer = text_model::ChineseClipTextTransformer::new(
        var.pp("text_model"),
        &text_model::ChineseClipTextConfig::clip_vit_base_patch16(),
    )?;
    println!("text_transformer: {:?}", text_transformer);

    Ok(())
}
