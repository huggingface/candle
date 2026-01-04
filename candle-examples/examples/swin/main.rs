//! Swin Transformer v1 image classification example.
//!
//! This example demonstrates using the Swin Transformer model for ImageNet classification.
//!
//! ```bash
//! cargo run --example swin --release -- --image path/to/image.jpg
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};

use candle::{DType, IndexOp, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::swin;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Version {
    V1,
    V2,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    Tiny,
    Small,
    Base,
    BaseLarge,
    Large,
    LargeLarge,
}

impl Which {
    fn model_id(&self, version: Version) -> String {
        match version {
            Version::V1 => {
                let name = match self {
                    Self::Tiny => "swin-tiny-patch4-window7-224",
                    Self::Small => "swin-small-patch4-window7-224",
                    Self::Base => "swin-base-patch4-window7-224",
                    Self::BaseLarge => "swin-base-patch4-window12-384",
                    Self::Large => "swin-large-patch4-window7-224",
                    Self::LargeLarge => "swin-large-patch4-window12-384",
                };
                format!("microsoft/{name}")
            }
            Version::V2 => {
                todo!("Swin v2 model IDs not yet implemented")
            }
        }
    }

    fn config(&self, version: Version) -> swin::Config {
        match version {
            Version::V1 => match self {
                Self::Tiny => swin::Config::tiny_224(),
                Self::Small => swin::Config::small_224(),
                Self::Base => swin::Config::base_224(),
                Self::BaseLarge => swin::Config::base_384(),
                Self::Large => swin::Config::large_224(),
                Self::LargeLarge => swin::Config::large_384(),
            },
            Version::V2 => {
                todo!("Swin v2 configs not yet implemented")
            }
        }
    }

    fn image_size(&self, version: Version) -> usize {
        match version {
            Version::V1 => match self {
                Self::BaseLarge | Self::LargeLarge => 384,
                _ => 224,
            },
            Version::V2 => {
                todo!("Swin v2 image sizes not yet implemented")
            }
        }
    }
}

#[derive(Parser)]
struct Args {
    /// Path to local model weights. If not provided, downloads from HuggingFace Hub.
    #[arg(long)]
    model: Option<String>,

    /// Path to the input image.
    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Model variant to use.
    #[arg(value_enum, long, default_value_t = Which::Tiny)]
    which: Which,

    /// Swin version (v1 or v2).
    #[arg(value_enum, long, default_value_t = Version::V1)]
    version: Version,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;

    let image_size = args.which.image_size(args.version);
    let image =
        candle_examples::imagenet::load_image(&args.image, image_size)?.to_device(&device)?;
    println!("loaded image {image:?}");

    let model_file = match args.model {
        None => {
            let model_id = args.which.model_id(args.version);
            println!("downloading model from {model_id}");
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model(model_id);
            // Try safetensors first, fall back to pytorch bin
            match api.get("model.safetensors") {
                Ok(p) => p,
                Err(_) => api.get("pytorch_model.bin")?,
            }
        }
        Some(model) => model.into(),
    };

    let vb = if model_file.extension().is_some_and(|ext| ext == "bin") {
        println!("loading pytorch weights from {:?}", model_file);
        VarBuilder::from_pth(&model_file, DType::F32, &device)?
    } else {
        println!("loading safetensors weights from {:?}", model_file);
        unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? }
    };

    let config = args.which.config(args.version);
    let model = swin::swin(&config, vb)?;
    println!("model built");

    let logits = model.forward(&image.unsqueeze(0)?)?;
    let prs = candle_nn::ops::softmax(&logits, D::Minus1)?
        .i(0)?
        .to_vec1::<f32>()?;

    let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
    prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));

    println!("\nTop-5 predictions:");
    for &(category_idx, pr) in prs.iter().take(5) {
        println!(
            "{:24}: {:.2}%",
            candle_examples::imagenet::CLASSES[category_idx],
            100. * pr
        );
    }

    Ok(())
}
