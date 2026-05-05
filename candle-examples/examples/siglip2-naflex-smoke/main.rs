// Real end-to-end test: load siglip2-base-patch16-naflex, preprocess real
// images into flattened patches at the model's base 16x16 grid (256 patches),
// tokenize text, run forward, check classification probabilities.

use anyhow::Result;
use candle::{DType, Device, Tensor, D};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::siglip2_naflex;
use tokenizers::Tokenizer;

const PATCH_SIZE: usize = 16;
const TARGET_GRID: usize = 16;
const IMAGE_SIZE: usize = PATCH_SIZE * TARGET_GRID; // 256

fn load_image_as_patches(path: &str, device: &Device) -> Result<Tensor> {
    use image::ImageReader;
    let img = ImageReader::open(path)?.decode()?;
    let img = img.resize_to_fill(
        IMAGE_SIZE as u32,
        IMAGE_SIZE as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8();
    let raw = img.into_raw();
    let img_tensor = Tensor::from_vec(raw, (IMAGE_SIZE, IMAGE_SIZE, 3), device)?
        .to_dtype(DType::F32)?
        .affine(2.0 / 255.0, -1.0)?;
    let g = TARGET_GRID;
    let p = PATCH_SIZE;
    let img_tensor = img_tensor
        .reshape((g, p, g, p, 3))?
        .permute((0, 2, 1, 3, 4))?
        .contiguous()?
        .reshape((g * g, p * p * 3))?;
    Ok(img_tensor.unsqueeze(0)?)
}

fn tokenize_pad(tokenizer: &Tokenizer, text: &str, pad_id: u32, max_len: usize) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut ids = encoding.get_ids().to_vec();
    if ids.len() > max_len {
        ids.truncate(max_len);
    } else {
        while ids.len() < max_len {
            ids.push(pad_id);
        }
    }
    Ok(ids)
}

fn main() -> Result<()> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model("google/siglip2-base-patch16-naflex".to_string());
    let model_file = repo.get("model.safetensors")?;
    let config_file = repo.get("config.json")?;
    let tokenizer_file = repo.get("tokenizer.json")?;

    let config: siglip2_naflex::Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(|e| anyhow::anyhow!("tok: {e}"))?;

    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let model = siglip2_naflex::Model::new(&config, vb)?;
    println!("Model loaded.");

    let bear = load_image_as_patches("/tmp/siglip2-practice-run/test_images/bear.jpg", &device)?;
    let teddy = load_image_as_patches("/tmp/siglip2-practice-run/test_images/teddy.jpg", &device)?;
    let images = Tensor::cat(&[&bear, &teddy], 0)?;
    println!("Images shape: {:?}", images.shape());

    let texts = vec![
        "a bear in the woods",
        "a robot holding a candle",
        "a group of teddy bears",
    ];
    let pad_id = config.text_config.pad_token_id;
    let max_len = config.text_config.max_position_embeddings;
    let mut all_ids: Vec<u32> = Vec::new();
    for t in &texts {
        all_ids.extend(tokenize_pad(&tokenizer, t, pad_id, max_len)?);
    }
    let input_ids = Tensor::from_vec(all_ids, (texts.len(), max_len), &device)?
        .to_dtype(DType::I64)?;
    println!("Input IDs shape: {:?}", input_ids.shape());

    let (logits_per_text, _logits_per_image) = model.forward(
        &images,
        None,
        None,
        Some((TARGET_GRID, TARGET_GRID)),
        &input_ids,
    )?;
    println!("logits_per_text shape: {:?}", logits_per_text.shape());
    let sig = candle_nn::ops::sigmoid(&logits_per_text)?;
    println!("Sigmoid scores (text x image):");
    let scores: Vec<Vec<f32>> = sig.to_vec2()?;
    for (i, t) in texts.iter().enumerate() {
        println!("  '{}' -> bear={:.4}, teddy={:.4}", t, scores[i][0], scores[i][1]);
    }
    println!("\nPer-image best match:");
    let logits_per_image = logits_per_text.t()?.contiguous()?;
    let probs = softmax(&logits_per_image, D::Minus1)?;
    let probs_v: Vec<Vec<f32>> = probs.to_vec2()?;
    let img_names = ["bear.jpg", "teddy.jpg"];
    for (i, name) in img_names.iter().enumerate() {
        let row = &probs_v[i];
        let best = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        println!("  {}: best='{}' prob={:.4}", name, texts[best.0], best.1);
    }

    Ok(())
}
