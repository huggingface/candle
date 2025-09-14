#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "cuda")]
use candle::CudaDevice;
use clap::Parser;

use candle::{BackendStorage, DType, IndexOp, Tensor, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::convmixer;

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

fn run<B: BackendStorage + 'static>(args: Args, device: &B::Device) -> anyhow::Result<()> {
    let image: Tensor<B> = candle_examples::imagenet::load_image224(args.image, device)?;
    println!("loaded image {image:?}");

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/candle-convmixer".into());
            api.get("convmixer_1024_20_ks9_p14.safetensors")?
        }
        Some(model) => model.into(),
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, device)? };
    let model = convmixer::c1024_20(1000, vb)?;
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

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.cpu {
        run::<candle::CpuStorage>(args, &candle::CpuDevice)?;
    } else {
        #[cfg(feature = "cuda")]
        run::<candle::CudaStorage>(args, &candle::CudaDevice::new(0)?)?;

        #[cfg(feature = "metal")]
        run::<candle::MetalStorage>(args, &candle::MetalDevice::new(0)?)?;
    }
    Ok(())
}
