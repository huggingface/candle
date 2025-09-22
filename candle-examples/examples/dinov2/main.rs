//! DINOv2: Learning Robust Visual Features without Supervision
//! https://github.com/facebookresearch/dinov2

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;

use candle::{BackendDevice, BackendStorage, DType, IndexOp, Tensor, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::dinov2;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
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

fn run<B: BackendStorage>(args: Args) -> anyhow::Result<()> {
    let device = B::Device::new(0)?;
    let image: Tensor<B> = candle_examples::imagenet::load_image224(args.image, &device)?;
    println!("loaded image {image:?}");

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/candle-dino-v2".into());
            api.get("dinov2_vits14.safetensors")?
        }
        Some(model) => model.into(),
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let model = dinov2::vit_small(vb)?;
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
