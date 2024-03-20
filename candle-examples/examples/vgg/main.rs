#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{DType, IndexOp, D};
use candle_nn::{ModuleT, VarBuilder};
use candle_transformers::models::vgg::{Models, Vgg};
use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    Vgg13,
    Vgg16,
    Vgg19,
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Variant of the model to use.
    #[arg(value_enum, long, default_value_t = Which::Vgg13)]
    which: Which,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let image = candle_examples::imagenet::load_image224(args.image)?.to_device(&device)?;

    println!("loaded image {image:?}");

    let api = hf_hub::api::sync::Api::new()?;
    let repo = match args.which {
        Which::Vgg13 => "timm/vgg13.tv_in1k",
        Which::Vgg16 => "timm/vgg16.tv_in1k",
        Which::Vgg19 => "timm/vgg19.tv_in1k",
    };
    let api = api.model(repo.into());
    let filename = "model.safetensors";
    let model_file = api.get(filename)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let model = match args.which {
        Which::Vgg13 => Vgg::new(vb, Models::Vgg13)?,
        Which::Vgg16 => Vgg::new(vb, Models::Vgg16)?,
        Which::Vgg19 => Vgg::new(vb, Models::Vgg19)?,
    };
    let logits = model.forward_t(&image, /*train=*/ false)?;

    let prs = candle_nn::ops::softmax(&logits, D::Minus1)?
        .i(0)?
        .to_vec1::<f32>()?;

    // Sort the predictions and take the top 5
    let mut top: Vec<_> = prs.iter().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    let top = top.into_iter().take(5).collect::<Vec<_>>();

    // Print the top predictions
    for &(i, p) in &top {
        println!(
            "{:50}: {:.2}%",
            candle_examples::imagenet::CLASSES[i],
            p * 100.0
        );
    }

    Ok(())
}
