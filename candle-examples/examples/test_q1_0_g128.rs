// Test: verify Q1_0_g128 tensors can be found and loaded from Bonsai-8B GGUF
// Run: cargo run --example test_q1_0_g128 --release
//
// This only loads tensor METADATA (headers), not full weight data.
// Total memory needed: ~50MB

use candle_core::quantized::gguf_file;
use candle_core::{Device, Result};
use std::fs::File;

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/zach/models/Bonsai-8B.gguf".to_string());

    println!("Loading: {}", model_path);
    println!();

    let device = Device::Cpu;
    let mut file = File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;

    println!("GGUF version: {:?}", content.magic);
    println!("Total tensors: {}", content.tensor_infos.len());
    println!();

    // Categorize all tensors by dtype
    let mut q1_0_g128_tensors = Vec::new();
    let mut other_dtypes: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for (name, info) in &content.tensor_infos {
        let dtype_name = format!("{:?}", info.ggml_dtype);
        if matches!(
            info.ggml_dtype,
            candle_core::quantized::GgmlDType::Q1_0_g128
        ) {
            q1_0_g128_tensors.push((name.clone(), info.shape.clone()));
        } else {
            *other_dtypes.entry(dtype_name).or_insert(0) += 1;
        }
    }

    println!("=== Non-Q1_0_g128 dtypes ===");
    for (dtype, count) in other_dtypes.iter().sorted_by(|a, b| b.1.cmp(a.1)) {
        println!("  {}: {} tensors", dtype, count);
    }
    println!();

    println!("=== Q1_0_g128 tensors ({}) ===", q1_0_g128_tensors.len());
    if q1_0_g128_tensors.is_empty() {
        println!("  NONE FOUND — check if this is actually a Q1_0_g128 GGUF!");
        return Err(candle_core::Error::Msg(
            "No Q1_0_g128 tensors found".to_string(),
        ));
    }
    for (name, shape) in &q1_0_g128_tensors {
        println!("  ✓ {}: shape={:?}", name, shape);
    }
    println!();

    // Verify we can read the raw tensor bytes for one tensor
    if let Some((name, _)) = q1_0_g128_tensors.first() {
        let info = &content.tensor_infos[name];
        println!("=== Attempting to load first Q1_0_g128 tensor ===");
        println!("  Tensor: {}", name);
        match info.read(&mut file, content.tensor_data_offset, &device) {
            Ok(qtensor) => {
                println!("  ✓ Loaded successfully!");
                println!("    shape: {:?}", qtensor.shape());
                println!("    dtype: {:?}", qtensor.dtype());
            }
            Err(e) => {
                println!("  ✗ FAILED to load: {}", e);
                return Err(e);
            }
        }
    }

    println!();
    println!("=== RESULT ===");
    println!("Q1_0_g128 tensors found: {}", q1_0_g128_tensors.len());
    println!("Q1_0_g128 loading: WORKING");
    println!();
    println!("SUCCESS: Q1_0_g128 support is functional!");
    Ok(())
}
