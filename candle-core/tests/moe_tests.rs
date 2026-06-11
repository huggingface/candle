use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Device, Result, Tensor,
};

#[test]
fn test_moe_gguf() -> Result<()> {
    let device = Device::new_cuda(0)?;

    // Setup MoE parameters
    let num_experts = 2;
    let n = 64;
    let k = 64;
    let batch_size = 2;
    let topk = 1;

    // 1. Create Expert Weights [num_experts, n, k]
    // We'll use Q8_0 quantization for the weights
    let weights_f32 = Tensor::randn(0.0f32, 1.0f32, (num_experts, n, k), &Device::Cpu)?;
    let qweights = QTensor::quantize_onto(&weights_f32, GgmlDType::Q8_0, &device)?;

    // 2. Create Input [batch_size, k]
    let input = Tensor::randn(0.0f32, 1.0f32, (batch_size, k), &device)?;

    // 3. Create Expert Indices [batch_size, topk]
    let ids = Tensor::from_slice(&[0u32, 1u32], (batch_size, topk), &device)?;

    // 4. Run MoE Forward
    // Note: indexed_moe_forward expects input of shape [batch, topk or 1, k]
    // or just [batch, k] depending on implementation details in candle-nn/candle-core
    // In QTensor::indexed_moe_forward in candle-core/src/quantized/mod.rs:655
    // it passes layouts directly to the backend.

    let input_moe = input.reshape((batch_size, 1, k))?;
    let output = qweights.indexed_moe_forward(&input_moe, &ids)?;

    // 5. Verify Output
    assert_eq!(output.dims(), &[batch_size, topk, n]);

    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    println!("MoE GGUF Output Sample: {:?}", &output_vec[..8]);

    Ok(())
}

/// Test FP8 (F8E4M3) fallback operations via FP16/FP32 emulation on SM < 8.9
/// This test verifies that the ALLOW_LEGACY_FP8 build flag works correctly.
#[test]
fn test_fp8_fallback() -> Result<()> {
    let device = Device::new_cuda(0)?;

    // Create test data in F32
    let size = 128;
    let data_f32 = Tensor::from_slice(
        &(0..size)
            .map(|i| (i as f32) * 0.1 - 6.4)
            .collect::<Vec<_>>(),
        size,
        &device,
    )?;

    // Test 1: Cast F32 -> F8E4M3 -> F32 roundtrip
    // On SM 6.1, this uses the emulated path via ALLOW_LEGACY_FP8
    let data_f8 = data_f32.to_dtype(DType::F8E4M3)?;
    let data_roundtrip = data_f8.to_dtype(DType::F32)?;

    // F8E4M3 has limited precision, so we expect some loss
    // Range: ~[-448, 448], precision: ~0.0625 at magnitude 1.0
    let original = data_f32.to_vec1::<f32>()?;
    let roundtrip = data_roundtrip.to_vec1::<f32>()?;

    println!("FP8 Roundtrip Test:");
    println!("  Original[0..8]: {:?}", &original[..8]);
    println!("  Roundtrip[0..8]: {:?}", &roundtrip[..8]);

    // Verify roundtrip is approximately correct (within FP8 precision)
    for (o, r) in original.iter().zip(roundtrip.iter()) {
        let diff = (o - r).abs();
        // FP8 E4M3 has ~3 bits of mantissa, so expect ~12.5% relative error for small values
        // and absolute tolerance for values near zero
        assert!(
            diff < 0.5 || diff / o.abs().max(1.0) < 0.25,
            "FP8 roundtrip error too large: {} -> {}, diff = {}",
            o,
            r,
            diff
        );
    }

    // Test 2: Unary operations on F8E4M3
    let data_f8_pos =
        Tensor::from_slice(&[0.5f32, 1.0, 2.0, 4.0], 4, &device)?.to_dtype(DType::F8E4M3)?;

    // Copy operation (identity)
    let copied = data_f8_pos.clone();
    assert_eq!(copied.to_dtype(DType::F32)?.to_vec1::<f32>()?.len(), 4);

    // Neg operation
    let negated = data_f8_pos.neg()?;
    let neg_vals = negated.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    println!("  Neg result: {:?}", neg_vals);
    assert!(neg_vals.iter().all(|x| *x <= 0.0));

    // Test 3: Binary operations on F8E4M3
    let a_f8 = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4, &device)?.to_dtype(DType::F8E4M3)?;
    let b_f8 = Tensor::from_slice(&[0.5f32, 0.5, 0.5, 0.5], 4, &device)?.to_dtype(DType::F8E4M3)?;

    // Add
    let sum = (&a_f8 + &b_f8)?;
    let sum_vals = sum.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    println!("  Add result: {:?}", sum_vals);

    // Mul
    let prod = (&a_f8 * &b_f8)?;
    let prod_vals = prod.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    println!("  Mul result: {:?}", prod_vals);

    // Verify results are sensible
    assert!((sum_vals[0] - 1.5).abs() < 0.2, "Add result incorrect");
    assert!((prod_vals[0] - 0.5).abs() < 0.1, "Mul result incorrect");

    println!("FP8 fallback test passed on current GPU!");

    Ok(())
}
