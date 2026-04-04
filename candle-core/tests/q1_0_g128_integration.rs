// Integration test: verify Q1_0_g128 GGUF can be read
//
// Run: cargo test --test q1_0_g128_integration -- --nocapture
//
// This test ONLY reads GGUF metadata — no inference, no matmul.
// Memory needed: ~50MB

use candle_core::quantized::gguf_file;
use candle_core::{Device, Result};

#[test]
fn test_q1_0_g128_tensor_detection() -> Result<()> {
    let model_path = "/home/zach/models/Bonsai-8B.gguf";

    println!("Loading: {}", model_path);

    let device = Device::Cpu;
    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;

    println!("GGUF version: {:?}", content.magic);
    println!("Total tensors: {}", content.tensor_infos.len());

    // Count tensors by dtype
    let mut q1_0_g128_count = 0;
    let mut q1_0_g128_names = Vec::new();

    for (name, info) in &content.tensor_infos {
        if matches!(
            info.ggml_dtype,
            candle_core::quantized::GgmlDType::Q1_0_g128
        ) {
            q1_0_g128_count += 1;
            q1_0_g128_names.push(name.clone());
        }
    }

    println!();
    println!("Q1_0_g128 tensors found: {}", q1_0_g128_count);

    if q1_0_g128_count == 0 {
        println!("ERROR: No Q1_0_g128 tensors found!");
        return Err(candle_core::Error::Msg(
            "No Q1_0_g128 tensors found".to_string(),
        ));
    }

    println!("Sample tensor names:");
    for name in q1_0_g128_names.iter().take(5) {
        println!("  - {}", name);
    }

    println!();
    println!("✓ Q1_0_g128 tensor detection: PASS");

    Ok(())
}

#[test]
fn test_q1_0_g128_tensor_loading() -> Result<()> {
    let model_path = "/home/zach/models/Bonsai-8B.gguf";

    let device = Device::Cpu;
    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;

    // Find first Q1_0_g128 tensor
    let mut first_q1_0_tensor = None;
    for (name, info) in &content.tensor_infos {
        if matches!(
            info.ggml_dtype,
            candle_core::quantized::GgmlDType::Q1_0_g128
        ) {
            first_q1_0_tensor = Some(name.clone());
            break;
        }
    }

    let tensor_name = first_q1_0_tensor
        .ok_or_else(|| candle_core::Error::Msg("No Q1_0_g128 tensor found".to_string()))?;

    println!("Loading tensor: {}", tensor_name);

    let info = &content.tensor_infos[&tensor_name];
    let loaded = info.read(&mut file, content.tensor_data_offset, &device)?;

    println!(
        "Loaded! shape={:?}, dtype={:?}",
        loaded.shape(),
        loaded.dtype()
    );

    println!();
    println!("✓ Q1_0_g128 tensor loading: PASS");

    Ok(())
}
