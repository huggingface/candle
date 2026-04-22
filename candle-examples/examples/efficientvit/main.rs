#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};

use candle::{DType, IndexOp, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::efficientvit;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    M0,
    M1,
    M2,
    M3,
    M4,
    M5,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::M0 => "m0",
            Self::M1 => "m1",
            Self::M2 => "m2",
            Self::M3 => "m3",
            Self::M4 => "m4",
            Self::M5 => "m5",
        };
        format!("timm/efficientvit_{name}.r224_in1k")
    }

    fn config(&self) -> efficientvit::Config {
        match self {
            Self::M0 => efficientvit::Config::m0(),
            Self::M1 => efficientvit::Config::m1(),
            Self::M2 => efficientvit::Config::m2(),
            Self::M3 => efficientvit::Config::m3(),
            Self::M4 => efficientvit::Config::m4(),
            Self::M5 => efficientvit::Config::m5(),
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

    #[arg(value_enum, long, default_value_t=Which::M0)]
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
    let model = efficientvit::efficientvit(&args.which.config(), 1000, vb)?;
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
