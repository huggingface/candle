#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};

use candle::{DType, IndexOp, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::mobilenetv4;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    Small,
    Medium,
    Large,
    HybridMedium,
    HybridLarge,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::Small => "conv_small.e2400_r224",
            Self::Medium => "conv_medium.e500_r256",
            Self::HybridMedium => "hybrid_medium.ix_e550_r256",
            Self::Large => "conv_large.e600_r384",
            Self::HybridLarge => "hybrid_large.ix_e600_r384",
        };
        format!("timm/mobilenetv4_{name}_in1k")
    }

    fn resolution(&self) -> u32 {
        match self {
            Self::Small => 224,
            Self::Medium => 256,
            Self::HybridMedium => 256,
            Self::Large => 384,
            Self::HybridLarge => 384,
        }
    }
    fn config(&self) -> mobilenetv4::Config {
        match self {
            Self::Small => mobilenetv4::Config::small(),
            Self::Medium => mobilenetv4::Config::medium(),
            Self::HybridMedium => mobilenetv4::Config::hybrid_medium(),
            Self::Large => mobilenetv4::Config::large(),
            Self::HybridLarge => mobilenetv4::Config::hybrid_large(),
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

    #[arg(value_enum, long, default_value_t=Which::Small)]
    which: Which,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;

    let image =
        candle_examples::imagenet::load_image(args.image, args.which.resolution() as usize)?
            .to_device(&device)?;
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
    let model = mobilenetv4::mobilenetv4(&args.which.config(), 1000, vb)?;
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
