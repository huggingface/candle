//! DINOv2 reg4 finetuned on PlantCLEF 2024
//! https://arxiv.org/abs/2309.16588
//! https://huggingface.co/spaces/BVRA/PlantCLEF2024
//! https://zenodo.org/records/10848263

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;

use candle::{DType, IndexOp, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::dinov2reg4;

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

    let device = candle_examples::device(args.cpu)?;

    let image = candle_examples::imagenet::load_image518(args.image)?.to_device(&device)?;
    println!("loaded image {image:?}");

    let f_species_id_mapping = "candle-examples/examples/dinov2reg4/species_id_mapping.txt";
    let classes: Vec<String> = std::fs::read_to_string(f_species_id_mapping)
        .expect("missing classes file")
        .split('\n')
        .map(|s| s.to_string())
        .collect();

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api =
                api.model("vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights".into());
            api.get(
                "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all.safetensors",
            )?
        }
        Some(model) => model.into(),
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let model = dinov2reg4::vit_base(vb)?;
    println!("model built");
    let logits = model.forward(&image.unsqueeze(0)?)?;
    let prs = candle_nn::ops::softmax(&logits, D::Minus1)?
        .i(0)?
        .to_vec1::<f32>()?;
    let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
    prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for &(category_idx, pr) in prs.iter().take(5) {
        println!("{:24}: {:.2}%", classes[category_idx], 100. * pr);
    }
    Ok(())
}
