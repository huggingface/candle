use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::vibevoice::{Config, VibeVoiceForConditionalGeneration};
use std::path::Path;

#[test]
fn test_vibevoice_weight_loading() {
    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/tmp/vibevoice-1.5b");

    if !model_dir.join("config.json").exists() {
        eprintln!("Skipping: model weights not found at {}", model_dir.display());
        return;
    }

    // 1. Load and parse config
    let config_str = std::fs::read_to_string(model_dir.join("config.json")).unwrap();
    let cfg: Config = serde_json::from_str(&config_str).expect("Failed to parse config.json");
    println!("Config parsed OK: hidden_size={}", cfg.decoder_config.hidden_size);

    // 2. Gather safetensors files
    let index_str =
        std::fs::read_to_string(model_dir.join("model.safetensors.index.json")).unwrap();
    let index: serde_json::Value = serde_json::from_str(&index_str).unwrap();
    let weight_map = index["weight_map"].as_object().unwrap();
    let mut shard_files: Vec<String> = weight_map.values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    shard_files.sort();
    shard_files.dedup();
    let filenames: Vec<std::path::PathBuf> = shard_files
        .iter()
        .map(|f| model_dir.join(f))
        .collect();
    println!("Loading {} shard files", filenames.len());

    // 3. Build VarBuilder from safetensors (mmap)
    let device = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device)
            .expect("Failed to mmap safetensors")
    };

    // 4. Construct the model — this validates all weight paths
    let mut model = VibeVoiceForConditionalGeneration::new(&cfg, vb)
        .expect("Failed to load model weights");
    println!("SUCCESS: VibeVoiceForConditionalGeneration loaded all weights");

    // 5. Verify tensor shapes with a dummy forward pass
    let input_ids = candle::Tensor::new(&[1u32, 2, 3, 4], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap(); // (1, 4)
    let logits = model.forward(&input_ids, 0).unwrap();
    let logits_shape = logits.dims().to_vec();
    println!("Forward pass logits shape: {:?}", logits_shape);
    assert_eq!(logits_shape[0], 1, "batch size");
    assert_eq!(logits_shape[1], 4, "seq_len");
    assert_eq!(logits_shape[2], cfg.decoder_config.vocab_size, "vocab_size");
    println!("SUCCESS: tensor shapes verified");
}
