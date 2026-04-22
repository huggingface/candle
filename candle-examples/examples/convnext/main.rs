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
    Atto,
    Femto,
    Pico,
    Nano,
    Tiny,
    Small,
    Base,
    Large,
    AttoV2,
    FemtoV2,
    PicoV2,
    NanoV2,
    TinyV2,
    BaseV2,
    LargeV2,
    XLarge,
    Huge,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::Atto => "convnext_atto.d2_in1k",
            Self::Femto => "convnext_femto.d1_in1k",
            Self::Pico => "convnext_pico.d1_in1k",
            Self::Nano => "convnext_nano.d1h_in1k",
            Self::Tiny => "convnext_tiny.fb_in1k",
            Self::Small => "convnext_small.fb_in1k",
            Self::Base => "convnext_base.fb_in1k",
            Self::Large => "convnext_large.fb_in1k",
            Self::AttoV2 => "convnextv2_atto.fcmae_ft_in1k",
            Self::FemtoV2 => "convnextv2_femto.fcmae_ft_in1k",
            Self::PicoV2 => "convnextv2_pico.fcmae_ft_in1k",
            Self::NanoV2 => "convnextv2_nano.fcmae_ft_in1k",
            Self::TinyV2 => "convnextv2_tiny.fcmae_ft_in1k",
            Self::BaseV2 => "convnextv2_base.fcmae_ft_in1k",
            Self::LargeV2 => "convnextv2_large.fcmae_ft_in1k",
            Self::XLarge => "convnext_xlarge.fb_in22k_ft_in1k",
            Self::Huge => "convnextv2_huge.fcmae_ft_in1k",
        };

        format!("timm/{name}")
    }

    fn config(&self) -> convnext::Config {
        match self {
            Self::Atto | Self::AttoV2 => convnext::Config::atto(),
            Self::Femto | Self::FemtoV2 => convnext::Config::femto(),
            Self::Pico | Self::PicoV2 => convnext::Config::pico(),
            Self::Nano | Self::NanoV2 => convnext::Config::nano(),
            Self::Tiny | Self::TinyV2 => convnext::Config::tiny(),
            Self::Small => convnext::Config::small(),
            Self::Base | Self::BaseV2 => convnext::Config::base(),
            Self::Large | Self::LargeV2 => convnext::Config::large(),
            Self::XLarge => convnext::Config::xlarge(),
            Self::Huge => convnext::Config::huge(),
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
