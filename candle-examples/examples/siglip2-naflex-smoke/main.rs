// Quick smoke test: load siglip2-base-patch16-naflex weights into our new model and run forward
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
    println!("Config loaded: vision hidden={}, layers={}, num_patches={}, patch_size={}",
        config.vision_config.hidden_size,
        config.vision_config.num_hidden_layers,
        config.vision_config.num_patches,
        config.vision_config.patch_size,
    );
    println!("Text config: hidden={}, vocab={}, max_pos={}, projection={}",
        config.text_config.hidden_size,
        config.text_config.vocab_size,
        config.text_config.max_position_embeddings,
        config.text_config.projection_size,
    );

    let device = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)?
    };

    println!("Constructing Model...");
    let model = siglip2_naflex::Model::new(&config, vb)?;
    println!("Model constructed successfully!");

    // Synthetic input: 16x16 patches = 256 patches, each 16x16x3 = 768-dim
    let target_h = 16;
    let target_w = 16;
    let num_patches = target_h * target_w;
    let patch_dim = 3 * 16 * 16;
    let pixel_values = Tensor::zeros((1, num_patches, patch_dim), DType::F32, &device)?;
    let input_ids = Tensor::zeros((1, 64), DType::I64, &device)?;

    println!("Running forward...");
    let (logits_per_text, logits_per_image) = model.forward(
        &pixel_values,
        None,
        None,
        Some((target_h, target_w)),
        &input_ids,
    )?;
    println!("logits_per_text shape: {:?}", logits_per_text.shape());
    println!("logits_per_image shape: {:?}", logits_per_image.shape());
    println!("logits_per_text first values: {:?}", logits_per_text.flatten_all()?.to_vec1::<f32>()?);

    Ok(())
}
