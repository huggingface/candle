#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};

use candle::{BackendDevice, BackendStorage, DType, IndexOp, Tensor, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::hiera;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    Tiny,
    Small,
    Base,
    BasePlus,
    Large,
    Huge,
}

impl Which {
    fn model_filename(&self) -> String {
        let name = match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Base => "base",
            Self::BasePlus => "base_plus",
            Self::Large => "large",
            Self::Huge => "huge",
        };
        format!("timm/hiera_{name}_224.mae_in1k_ft_in1k")
    }

    fn config(&self) -> hiera::Config {
        match self {
            Self::Tiny => hiera::Config::tiny(),
            Self::Small => hiera::Config::small(),
            Self::Base => hiera::Config::base(),
            Self::BasePlus => hiera::Config::base_plus(),
            Self::Large => hiera::Config::large(),
            Self::Huge => hiera::Config::huge(),
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

    if args.cpu {
        run::<candle::CpuStorage>(args)?;
    } else if candle::utils::cuda_is_available() {
        run::<candle::CudaStorage>(args)?;
    } else if candle::utils::metal_is_available() {
        run::<candle::MetalStorage>(args)?;
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        run::<candle::CpuStorage>(args)?;
    }
    Ok(())
}

fn run<B: BackendStorage + 'static>(args: Args) -> anyhow::Result<()> {
    let device = B::Device::new(0)?;
    let image: Tensor<B> = candle_examples::imagenet::load_image224(args.image, &device)?;
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
    let model = hiera::hiera(&args.which.config(), 1000, vb)?;
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
