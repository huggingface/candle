#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};

use candle::{DType, IndexOp, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::repvgg;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    A0,
    A1,
    A2,
    B0,
    B1,
    B2,
    B3,
    B1G4,
    B2G4,
    B3G4,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::A0 => "a0",
            Self::A1 => "a1",
            Self::A2 => "a2",
            Self::B0 => "b0",
            Self::B1 => "b1",
            Self::B2 => "b2",
            Self::B3 => "b3",
            Self::B1G4 => "b1g4",
            Self::B2G4 => "b2g4",
            Self::B3G4 => "b3g4",
        };
        format!("timm/repvgg_{name}.rvgg_in1k")
    }

    fn config(&self) -> repvgg::Config {
        match self {
            Self::A0 => repvgg::Config::a0(),
            Self::A1 => repvgg::Config::a1(),
            Self::A2 => repvgg::Config::a2(),
            Self::B0 => repvgg::Config::b0(),
            Self::B1 => repvgg::Config::b1(),
            Self::B2 => repvgg::Config::b2(),
            Self::B3 => repvgg::Config::b3(),
            Self::B1G4 => repvgg::Config::b1g4(),
            Self::B2G4 => repvgg::Config::b2g4(),
            Self::B3G4 => repvgg::Config::b3g4(),
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

    #[arg(value_enum, long, default_value_t=Which::A0)]
    which: Which,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;

    let image = candle_examples::imagenet::load_image224(args.image)?.to_device(&device)?;
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
    let model = repvgg::repvgg(&args.which.config(), 1000, vb)?;
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
