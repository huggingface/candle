#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};

use candle::{DType, IndexOp, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::convnext;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    Tiny,
    Small,
    Base,
    Large,
    XLarge,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Base => "base",
            Self::Large => "large",
            Self::XLarge => "xlarge",
        };
        // The XLarge model only has an ImageNet-22K variant
        let variant = match self {
            Self::XLarge => "fb_in22k_ft_in1k",
            _ => "fb_in1k",
        };

        format!("timm/convnext_{name}.{variant}")
    }

    fn config(&self) -> convnext::Config {
        match self {
            Self::Tiny => convnext::Config::tiny(),
            Self::Small => convnext::Config::small(),
            Self::Base => convnext::Config::base(),
            Self::Large => convnext::Config::large(),
            Self::XLarge => convnext::Config::xlarge(),
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

    #[arg(value_enum, long, default_value_t=Which::Tiny)]
    which: Which,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;

    let image = candle_examples::imagenet::load_image224(args.image)?;
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
    let model = convnext::convnext(&args.which.config(), 1000, vb)?;
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
