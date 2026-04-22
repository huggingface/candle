#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{DType, IndexOp, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::resnet;
use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    #[value(name = "18")]
    Resnet18,
    #[value(name = "34")]
    Resnet34,
    #[value(name = "50")]
    Resnet50,
    #[value(name = "101")]
    Resnet101,
    #[value(name = "152")]
    Resnet152,
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

    /// Variant of the model to use.
    #[arg(value_enum, long, default_value_t = Which::Resnet18)]
    which: Which,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;

    let image = candle_examples::imagenet::load_image224(args.image)?.to_device(&device)?;
    println!("loaded image {image:?}");

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/candle-resnet".into());
            let filename = match args.which {
                Which::Resnet18 => "resnet18.safetensors",
                Which::Resnet34 => "resnet34.safetensors",
                Which::Resnet50 => "resnet50.safetensors",
                Which::Resnet101 => "resnet101.safetensors",
                Which::Resnet152 => "resnet152.safetensors",
            };
            api.get(filename)?
        }
        Some(model) => model.into(),
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let class_count = candle_examples::imagenet::CLASS_COUNT as usize;
    let model = match args.which {
        Which::Resnet18 => resnet::resnet18(class_count, vb)?,
        Which::Resnet34 => resnet::resnet34(class_count, vb)?,
        Which::Resnet50 => resnet::resnet50(class_count, vb)?,
        Which::Resnet101 => resnet::resnet101(class_count, vb)?,
        Which::Resnet152 => resnet::resnet152(class_count, vb)?,
    };
    println!("model built");
    let logits = model.forward(&image.unsqueeze(0)?)?;
    let prs = candle_nn::ops::softmax(&logits, D::Minus1)?
        .i(0)?
        .to_vec1::<f32>()?;
    let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
    prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for &(category_idx, pr) in prs.iter().take(5) {
        println!(
            "{:24}: {:.2}%",
            candle_examples::imagenet::CLASSES[category_idx],
            100. * pr
        );
    }
    Ok(())
}
