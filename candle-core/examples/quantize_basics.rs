use candle_core::{DType, Device, QuantizedDType, Result, Tensor};

fn main() -> Result<()> {
    // Create an f32 tensor with 256 elements (matches Q8_0 block size)
    let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1).collect();
    let tensor = Tensor::from_slice(&data, 256, &Device::Cpu)?;

    println!("Original tensor: F32");
    println!("First 10 values: {:?}\n", &tensor.to_vec1::<f32>()?[..10]);

    // Test different quantization types
    let qtypes = vec![
        ("Q4_0", QuantizedDType::GgmlQ4_0),
        ("Q4_1", QuantizedDType::GgmlQ4_1),
        ("Q5_0", QuantizedDType::GgmlQ5_0),
        ("Q5_1", QuantizedDType::GgmlQ5_1),
        ("Q8_0", QuantizedDType::GgmlQ8_0),
        // ("Q8_1", QuantizedDType::Q8_1), // Skipped - not fully implemented
        ("Q2K", QuantizedDType::GgmlQ2K),
        ("Q3K", QuantizedDType::GgmlQ3K),
        ("Q4K", QuantizedDType::GgmlQ4K),
        ("Q5K", QuantizedDType::GgmlQ5K),
        ("Q6K", QuantizedDType::GgmlQ6K),
        ("Q8K", QuantizedDType::GgmlQ8K),
    ];

    // Display backend support matrix
    println!("\n=== Backend Support Matrix ===\n");
    println!(
        "{:<6} | {:^6} | {:^6} | {:^6}",
        "Type", "CPU", "CUDA", "Metal"
    );
    println!("{:-<6}-+-{:-<6}-+-{:-<6}-+-{:-<6}", "", "", "", "");

    for (name, qtype) in &qtypes {
        let has_cpu = qtype.has_cpu();
        let has_cuda = qtype.has_cuda();
        let has_metal = qtype.has_metal();

        println!(
            "{:<6} | {:^6} | {:^6} | {:^6}",
            name,
            if has_cpu { "✓" } else { "✗" },
            if has_cuda { "✓" } else { "✗" },
            if has_metal { "✓" } else { "✗" }
        );
    }

    println!("\n=== Quantization Quality Comparison ===\n");
    println!(
        "{:<6} | {:>12} | {:>12} | {:>12}",
        "Type", "Avg Error", "Max Error", "Rel Error %"
    );
    println!("{:-<6}-+-{:-<12}-+-{:-<12}-+-{:-<12}", "", "", "", "");

    let original_data = tensor.to_vec1::<f32>()?;

    for (name, qtype) in qtypes {
        // Quantize
        let quantized = tensor.to_dtype(DType::Quantized(qtype))?;

        // Dequantize
        let dequantized = quantized.to_dtype(DType::F32)?;
        let recovered_data = dequantized.to_vec1::<f32>()?;

        // Calculate errors
        let mut total_error = 0.0f32;
        let mut max_error = 0.0f32;
        let mut total_rel_error = 0.0f32;
        let mut count = 0;

        for (orig, recovered) in original_data.iter().zip(recovered_data.iter()) {
            let error = (orig - recovered).abs();
            total_error += error;
            max_error = max_error.max(error);

            if orig.abs() > 1e-6 {
                total_rel_error += error / orig.abs();
                count += 1;
            }
        }

        let avg_error = total_error / original_data.len() as f32;
        let avg_rel_error = if count > 0 {
            (total_rel_error / count as f32) * 100.0
        } else {
            0.0
        };

        println!(
            "{:<6} | {:>12.6} | {:>12.6} | {:>11.2}%",
            name, avg_error, max_error, avg_rel_error
        );
    }

    println!("\n=== Detailed Example: Q8_0 vs Q4K ===\n");

    // Q8_0 (higher quality, 8-bit quantization)
    let q8_quantized = tensor.to_dtype(DType::Quantized(QuantizedDType::GgmlQ8_0))?;
    let q8_dequantized = q8_quantized.to_dtype(DType::F32)?;
    let q8_data = q8_dequantized.to_vec1::<f32>()?;

    // Q4K (more compressed, 4-bit quantization)
    let q4k_quantized = tensor.to_dtype(DType::Quantized(QuantizedDType::GgmlQ4K))?;
    let q4k_dequantized = q4k_quantized.to_dtype(DType::F32)?;
    let q4k_data = q4k_dequantized.to_vec1::<f32>()?;

    println!("Index | Original  | Q8_0      | Q4K       | Q8_0 Err  | Q4K Err");
    println!(
        "{:-<5}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}",
        "", "", "", "", "", ""
    );

    for i in (0..20).step_by(2) {
        let orig = original_data[i];
        let q8 = q8_data[i];
        let q4k = q4k_data[i];
        let q8_err = (orig - q8).abs();
        let q4k_err = (orig - q4k).abs();

        println!(
            "{:>5} | {:>9.5} | {:>9.5} | {:>9.5} | {:>9.5} | {:>9.5}",
            i, orig, q8, q4k, q8_err, q4k_err
        );
    }

    println!("\n=== Testing Operations with Quantized Tensors ===\n");

    // Input activations (f32) - shape [1, 256]
    let input_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let activations = Tensor::from_slice(&input_data, (1, 256), &Device::Cpu)?;

    // Create weight matrix (will be quantized) - shape [256, 128] (smaller for demo)
    let weight_data: Vec<f32> = (0..256 * 128).map(|i| (i as f32) * 0.001).collect();
    let weights_f32 = Tensor::from_slice(&weight_data, (256, 128), &Device::Cpu)?;

    println!("Input activations: F32, shape {:?}", activations.shape());
    println!("Weight matrix (F32): F32, shape {:?}", weights_f32.shape());

    // Quantize the weights (common pattern: keep activations in f32, quantize weights)
    let weights_q4k = weights_f32.to_dtype(DType::Quantized(QuantizedDType::GgmlQ4K))?;
    let weights_q8_0 = weights_f32.to_dtype(DType::Quantized(QuantizedDType::GgmlQ8_0))?;
    println!(
        "Weight matrix (Q4K): Quantized, shape {:?}",
        weights_q4k.shape()
    );
    println!(
        "Weight matrix (Q8_0): Quantized, shape {:?}\n",
        weights_q8_0.shape()
    );

    // Test 1: Matmul with f32 x quantized Q4K (specialized fast path)
    println!("Test 1: F32 x Quantized(Q4K) Matmul (specialized implementation)");
    let result_q4k = activations.matmul(&weights_q4k)?;
    println!(
        "  Result shape: {:?}, dtype: {:?}",
        result_q4k.shape(),
        result_q4k.dtype()
    );
    let result_q4k_values = result_q4k.to_vec2::<f32>()?;
    println!("  First 5 values: {:?}", &result_q4k_values[0][..5]);

    // Test 2: Matmul with f32 x quantized Q8_0
    println!("\nTest 2: F32 x Quantized(Q8_0) Matmul");
    let result_q8 = activations.matmul(&weights_q8_0)?;
    println!(
        "  Result shape: {:?}, dtype: {:?}",
        result_q8.shape(),
        result_q8.dtype()
    );
    let result_q8_values = result_q8.to_vec2::<f32>()?;
    println!("  First 5 values: {:?}", &result_q8_values[0][..5]);

    // Compare Q4K vs Q8_0 matmul results
    println!("\n  Matmul Comparison: Q4K vs Q8_0 (Q8_0 is more accurate):");
    println!(
        "  {:>5} | {:>12} | {:>12} | {:>10}",
        "Index", "Q4K", "Q8_0", "Diff"
    );
    println!("  {:-<5}-+-{:-<12}-+-{:-<12}-+-{:-<10}", "", "", "", "");
    for i in 0..5 {
        let q4k_val = result_q4k_values[0][i];
        let q8_val = result_q8_values[0][i];
        let diff = (q4k_val - q8_val).abs();
        println!(
            "  {:>5} | {:>12.3} | {:>12.3} | {:>10.3}",
            i, q4k_val, q8_val, diff
        );
    }

    // Test 3: Addition with automatic dequantization
    println!("\nTest 3: Quantized + Quantized Addition (auto-dequantize)");
    let q1 = tensor.to_dtype(DType::Quantized(QuantizedDType::GgmlQ8_0))?;
    let q2 = tensor.to_dtype(DType::Quantized(QuantizedDType::GgmlQ8_0))?;
    let sum = (&q1 + &q2)?; // Automatically dequantizes, adds, re-quantizes
    println!("  Result dtype: {:?}", sum.dtype());
    println!(
        "  First 5 values: {:?}",
        &sum.to_dtype(DType::F32)?.to_vec1::<f32>()?[..5]
    );

    // Test 4: Scalar multiplication
    println!("\nTest 4: Quantized Tensor * F32 Scalar");
    let scaled = (&q1 * 2.0)?; // Automatically dequantizes, scales, re-quantizes
    println!("  Result dtype: {:?}", scaled.dtype());
    let scaled_f32 = scaled.to_dtype(DType::F32)?;
    println!("  First 5 values: {:?}", &scaled_f32.to_vec1::<f32>()?[..5]);

    // Test 5: Mixed precision (F32 + Quantized)
    println!("\nTest 5: Mixed Precision (F32 + Quantized)");
    let f32_tensor = tensor.to_dtype(DType::F32)?;
    let mixed_sum = (&f32_tensor + &q1)?; // Auto-dequantizes quantized, returns F32
    println!("  Result dtype: {:?}", mixed_sum.dtype());
    println!("  First 5 values: {:?}", &mixed_sum.to_vec1::<f32>()?[..5]);

    // Test 6: Reshaping (works directly on quantized data)
    println!("\nTest 6: Reshaping Quantized Tensor");
    let reshaped = q1.reshape((16, 16))?;
    println!(
        "  Original shape: {:?}, New shape: {:?}",
        q1.shape(),
        reshaped.shape()
    );
    println!("  Dtype preserved: {:?}", reshaped.dtype());

    // Test 7: Conversion between quantization types
    println!("\nTest 7: Converting Between Quantization Types");
    let q4k = tensor.to_dtype(DType::Quantized(QuantizedDType::GgmlQ4K))?;
    let q8k = q4k.to_dtype(DType::Quantized(QuantizedDType::GgmlQ8K))?;
    println!("  Q4K → Q8K conversion");
    println!(
        "  Original dtype: {:?}, New dtype: {:?}",
        q4k.dtype(),
        q8k.dtype()
    );
    let q8k_values = q8k.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    println!("  First 5 values after conversion: {:?}", &q8k_values[..5]);

    Ok(())
}
