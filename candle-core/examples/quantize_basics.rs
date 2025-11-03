//! Quantization examples: demonstrating GGML quantization types and operations
//!
//! Run: `cargo run --example quantize_basics --release [--features cuda]`

use candle_core::{DType, Device, QuantizedDType, Result, Tensor};

fn main() -> Result<()> {
    println!("=== Candle Quantization Demo ===\n");

    #[cfg(feature = "cuda")]
    let cuda_device = Device::new_cuda(0).ok();
    #[cfg(not(feature = "cuda"))]
    let _cuda_device: Option<Device> = None;

    // Create test tensor
    let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1).collect();
    let tensor = Tensor::from_slice(&data, 256, &Device::Cpu)?;

    // Test quantization types
    let qtypes = [
        ("Q2K", QuantizedDType::GgmlQ2K, 2.6),
        ("Q4K", QuantizedDType::GgmlQ4K, 4.0),
        ("Q8_0", QuantizedDType::GgmlQ8_0, 8.0),
    ];

    println!("1. Accuracy Comparison\n");
    println!("Type  | Bits | Error");
    println!("------+------+-------");

    let original_data = tensor.to_vec1::<f32>()?;

    for (name, qtype, bits) in &qtypes {
        let quantized = tensor.to_dtype(DType::Quantized(*qtype))?;
        let recovered = quantized.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let error = original_data
            .iter()
            .zip(&recovered)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / original_data.len() as f32;
        println!("{:<5} | {:>4.1} | {:.4}", name, bits, error);
    }

    // Quantized operations
    println!("\n2. Quantized Operations (CPU)\n");

    // Addition (Q8_0 block size = 32)
    let a = Tensor::from_slice(
        &(0..32).map(|i| i as f32).collect::<Vec<_>>(),
        32,
        &Device::Cpu,
    )?;
    let b = Tensor::from_slice(
        &(0..32).map(|i| (i as f32) * 0.5).collect::<Vec<_>>(),
        32,
        &Device::Cpu,
    )?;
    let a_q = a.to_dtype(DType::Quantized(QuantizedDType::GgmlQ8_0))?;
    let b_q = b.to_dtype(DType::Quantized(QuantizedDType::GgmlQ8_0))?;
    let sum = (&a_q + &b_q)?;
    let sum_result = sum.to_vec1::<f32>()?;
    println!(
        "✓ Add: Q8_0[32] + Q8_0[32] → [{:.1}, {:.1}, {:.1}, ...]",
        sum_result[0], sum_result[1], sum_result[2]
    );

    // Multiplication
    let mul = (&a_q * &b_q)?;
    let mul_result = mul.to_vec1::<f32>()?;
    println!(
        "✓ Mul: Q8_0[32] * Q8_0[32] → [{:.1}, {:.1}, {:.1}, ...]",
        mul_result[0], mul_result[1], mul_result[2]
    );

    // MatMul
    let weights = Tensor::from_slice(
        &(0..256 * 64)
            .map(|i| (i as f32) * 0.001)
            .collect::<Vec<_>>(),
        (256, 64),
        &Device::Cpu,
    )?;
    let activations = Tensor::from_slice(
        &(0..256).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
        (1, 256),
        &Device::Cpu,
    )?;
    let weights_q4k = weights.to_dtype(DType::Quantized(QuantizedDType::GgmlQ4K))?;
    let result = activations.matmul(&weights_q4k)?;
    println!("✓ MatMul: F32[1,256] × Q4K[256,64] → {:?}", result.shape());

    // CUDA examples
    #[cfg(feature = "cuda")]
    if let Some(ref dev) = cuda_device {
        if let Err(e) = cuda_examples(dev) {
            println!("⚠ CUDA error: {}", e);
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_examples(device: &Device) -> Result<()> {
    println!("\n3. CUDA GPU Acceleration\n");

    // Roundtrip test
    let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05).collect();
    let gpu_tensor = Tensor::from_slice(&data, 256, device)?;
    let gpu_q4k = gpu_tensor.to_dtype(DType::Quantized(QuantizedDType::GgmlQ4K))?;
    let gpu_deq = gpu_q4k.to_dtype(DType::F32)?;
    let result = gpu_deq.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let error = data
        .iter()
        .zip(&result)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / data.len() as f32;
    println!("✓ Q4K roundtrip error: {:.6}", error);

    // GPU matmul
    let weights = Tensor::from_slice(
        &(0..128 * 64)
            .map(|i| (i as f32) * 0.001)
            .collect::<Vec<_>>(),
        (128, 64),
        device,
    )?;
    let activations = Tensor::from_slice(
        &(0..128).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
        (1, 128),
        device,
    )?;
    let weights_q4k = weights.to_dtype(DType::Quantized(QuantizedDType::GgmlQ4K))?;
    let result = activations.matmul(&weights_q4k)?;

    let result_cpu = result.to_device(&Device::Cpu)?;
    let result_vec = result_cpu.to_vec2::<f32>()?;
    println!(
        "✓ GPU MatMul: {:?} → {:?}",
        activations.shape(),
        result_cpu.shape()
    );
    println!(
        "First 3 values: [{:.4}, {:.4}, {:.4}]",
        result_vec[0][0], result_vec[0][1], result_vec[0][2]
    );

    Ok(())
}
