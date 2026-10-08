// Test forward_variable with explicit spatial_shapes (true NaFlex per-input variation)
// Two images, different spatial shapes per input, padded to max_num_patches.
use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::siglip2_naflex;

fn main() -> Result<()> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model("google/siglip2-base-patch16-naflex".to_string());
    let model_file = repo.get("model.safetensors")?;
    let config_file = repo.get("config.json")?;

    let config: siglip2_naflex::Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let model = siglip2_naflex::Model::new(&config, vb)?;
    println!("Model loaded.");

    // Two images: image 0 is 16x16 (256 patches, full base grid), image 1 is 12x12 (144 patches, smaller).
    // Pad image 1 to 256 patches with zeros, mask the padded positions.
    let patch_dim = 3 * 16 * 16; // 768
    let max_n = 256;

    // Input 0: full 16x16 grid, all real patches with random values
    let img0 = Tensor::randn(0.0f32, 1.0, (max_n, patch_dim), &device)?;
    // Input 1: 12x12 = 144 real patches, then 112 zero-padded
    let img1_real = Tensor::randn(0.0f32, 1.0, (144, patch_dim), &device)?;
    let img1_pad = Tensor::zeros((max_n - 144, patch_dim), DType::F32, &device)?;
    let img1 = Tensor::cat(&[&img1_real, &img1_pad], 0)?;
    let pixel_values = Tensor::stack(&[&img0, &img1], 0)?; // (2, 256, 768)

    // pixel_attention_mask: (2, 256), 1 for real, 0 for pad
    let mask0 = Tensor::ones(max_n, DType::F32, &device)?;
    let mask1_real = Tensor::ones(144, DType::F32, &device)?;
    let mask1_pad = Tensor::zeros(max_n - 144, DType::F32, &device)?;
    let mask1 = Tensor::cat(&[&mask1_real, &mask1_pad], 0)?;
    let pixel_attention_mask = Tensor::stack(&[&mask0, &mask1], 0)?; // (2, 256)

    // spatial_shapes: (2, 2) - [[16,16],[12,12]]
    let spatial_shapes = Tensor::from_vec(vec![16i64, 16, 12, 12], (2, 2), &device)?;

    let input_ids = Tensor::zeros((1, 64), DType::I64, &device)?;

    println!("Running variable-shape forward...");
    let (logits, _) = model.forward(
        &pixel_values,
        Some(&pixel_attention_mask),
        Some(&spatial_shapes),
        None,
        &input_ids,
    )?;
    println!("logits shape: {:?}", logits.shape());
    println!(
        "logits values: {:?}",
        logits.flatten_all()?.to_vec1::<f32>()?
    );
    Ok(())
}
