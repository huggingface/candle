#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};

use candle::{DType, IndexOp, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::fastvit;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    T8,
    T12,
    S12,
    SA12,
    SA24,
    SA36,
    MA36,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::T8 => "t8",
            Self::T12 => "t12",
            Self::S12 => "s12",
            Self::SA12 => "sa12",
            Self::SA24 => "sa24",
            Self::SA36 => "sa36",
            Self::MA36 => "ma36",
        };
        format!("timm/fastvit_{name}.apple_in1k")
    }

    fn config(&self) -> fastvit::Config {
        match self {
            Self::T8 => fastvit::Config::t8(),
            Self::T12 => fastvit::Config::t12(),
            Self::S12 => fastvit::Config::s12(),
            Self::SA12 => fastvit::Config::sa12(),
            Self::SA24 => fastvit::Config::sa24(),
            Self::SA36 => fastvit::Config::sa36(),
            Self::MA36 => fastvit::Config::ma36(),
        }
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

    #[arg(value_enum, long, default_value_t=Which::S12)]
    which: Which,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;

    let image = candle_examples::imagenet::load_image(args.image, 256)?.to_device(&device)?;
    println!("loaded image {image:?}");

    let model_file = match args.model {
        None => {
            let model_name = args.which.model_filename();
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model(model_name);
            api.get("model.safetensors")?
        }
        Some(model) => model.into(),
    };

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let model = fastvit::fastvit(&args.which.config(), 1000, vb)?;
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
