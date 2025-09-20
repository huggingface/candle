//! BEiT: BERT Pre-Training of Image Transformers
//! https://github.com/microsoft/unilm/tree/master/beit

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{BackendDevice, BackendStorage, DType, IndexOp, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::beit;
use clap::Parser;

/// Loads an image from disk using the image crate, this returns a tensor with shape
/// (3, 384, 384). Beit special normalization is applied.
pub fn load_image384_beit_norm<P: AsRef<std::path::Path>, B: BackendStorage>(
    p: P,
    device: &B::Device,
) -> Result<Tensor<B>> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?
        .resize_to_fill(384, 384, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (384, 384, 3), device)?.permute((2, 0, 1))?;
    let mean = Tensor::new(&[0.5f32, 0.5, 0.5], device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.5f32, 0.5, 0.5], device)?.reshape((3, 1, 1))?;
    (data.to_dtype(candle::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
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
}

fn run<B: BackendStorage>(args: Args) -> anyhow::Result<()> {
    let device = B::Device::new(0)?;
    let image: Tensor<B> = load_image384_beit_norm(args.image, &device)?;
    println!("loaded image {image:?}");

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("vincent-espitalier/candle-beit".into());
            api.get("beit_base_patch16_384.in22k_ft_in22k_in1k.safetensors")?
        }
        Some(model) => model.into(),
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let model = beit::vit_base(vb)?;
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
