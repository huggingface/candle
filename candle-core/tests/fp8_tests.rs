use candle_core::{test_device, DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
fn uses_legacy_fp8(device: &Device) -> bool {
    device.is_cuda() && candle_core::cuda_backend::kernels::capabilities::ALLOW_LEGACY_FP8
}

#[cfg(not(feature = "cuda"))]
fn uses_legacy_fp8(_device: &Device) -> bool {
    false
}

/// Test cast roundtrip: F32 -> F8E4M3 -> F32
fn cast_fp8(device: &Device) -> Result<()> {
    if device.is_cpu() {
        return Ok(());
    }
    let data = &[
        -16.0f32, -2.0, -1.0, -0.5, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0,
    ];
    let t = Tensor::new(data, device)?;
    let fp8 = t.to_dtype(DType::F8E4M3)?;
    let back = fp8.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let abs_tol = if uses_legacy_fp8(device) { 0.5 } else { 0.25 };
    let rel_tol = if uses_legacy_fp8(device) { 0.25 } else { 0.125 };

    for (orig, round) in data.iter().zip(back.iter()) {
        let diff = (orig - round).abs();
        assert!(
            diff < abs_tol || diff / orig.abs().max(1.0) < rel_tol,
            "FP8 cast roundtrip error too large: {} -> {}, diff = {}",
            orig,
            round,
            diff
        );
        assert!(!round.is_nan(), "NaN in FP8 cast roundtrip: {}", round);
        assert!(!round.is_infinite(), "Inf in FP8 cast roundtrip: {}", round);
    }
    println!("FP8 cast roundtrip OK");
    Ok(())
}

/// Test unary operations on F8E4M3
fn unary_fp8(device: &Device) -> Result<()> {
    if device.is_cpu() {
        return Ok(());
    }
    let data = &[0.5f32, 1.0, 2.0, 4.0];
    let t = Tensor::new(data, device)?.to_dtype(DType::F8E4M3)?;

    // Neg
    let neg = t.neg()?;
    let neg_vals = neg.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    for v in &neg_vals {
        assert!(
            *v <= 0.0,
            "neg should produce non-positive values, got {}",
            v
        );
        assert!(!v.is_nan(), "NaN in neg result");
    }

    // Recip
    let recip = t.recip()?;
    let recip_vals = recip.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let expected_recip = [2.0f32, 1.0, 0.5, 0.25];
    for (act, exp) in recip_vals.iter().zip(expected_recip.iter()) {
        assert!(
            (act - exp).abs() < 0.2,
            "recip mismatch: {} vs expected {}",
            act,
            exp
        );
        assert!(!act.is_nan(), "NaN in recip result");
    }

    // Exp
    let exp_t = t.exp()?;
    let exp_vals = exp_t.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    for v in &exp_vals {
        assert!(*v > 0.0, "exp should be positive, got {}", v);
        assert!(!v.is_nan(), "NaN in exp result");
    }

    // Sqrt
    let sqrt_t = t.sqrt()?;
    let sqrt_vals = sqrt_t.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    for v in &sqrt_vals {
        assert!(*v > 0.0, "sqrt should be positive, got {}", v);
        assert!(!v.is_nan(), "NaN in sqrt result");
    }

    println!(
        "FP8 unary ops OK: neg={:?}, recip={:?}, exp={:?}, sqrt={:?}",
        neg_vals, recip_vals, exp_vals, sqrt_vals
    );
    Ok(())
}

/// Test binary operations on F8E4M3
fn binary_fp8(device: &Device) -> Result<()> {
    if device.is_cpu() {
        return Ok(());
    }
    let a_data = &[-2.0f32, -1.0, 2.0, 4.0];
    let b_data = &[0.5f32, -0.5, 0.5, 2.0];
    let binary_tol = if uses_legacy_fp8(device) { 0.5 } else { 0.25 };
    let div_tol = if uses_legacy_fp8(device) { 1.0 } else { 0.5 };

    let a = Tensor::new(a_data, device)?.to_dtype(DType::F8E4M3)?;
    let b = Tensor::new(b_data, device)?.to_dtype(DType::F8E4M3)?;

    // Add
    let c_add = a.broadcast_add(&b)?;
    let v_add = c_add.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    for (act, exp) in v_add.iter().zip([-1.5f32, -1.5, 2.5, 6.0].iter()) {
        assert!(
            (act - exp).abs() < binary_tol,
            "add mismatch: {} vs {}",
            act,
            exp
        );
        assert!(!act.is_nan(), "NaN in add");
        assert!(!act.is_infinite(), "Inf in add");
    }

    // Sub
    let c_sub = a.broadcast_sub(&b)?;
    let v_sub = c_sub.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    for (act, exp) in v_sub.iter().zip([-2.5f32, -0.5, 1.5, 2.0].iter()) {
        assert!(
            (act - exp).abs() < binary_tol,
            "sub mismatch: {} vs {}",
            act,
            exp
        );
        assert!(!act.is_nan(), "NaN in sub");
    }

    // Mul
    let c_mul = a.broadcast_mul(&b)?;
    let v_mul = c_mul.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    for (act, exp) in v_mul.iter().zip([-1.0f32, 0.5, 1.0, 8.0].iter()) {
        assert!(
            (act - exp).abs() < binary_tol,
            "mul mismatch: {} vs {}",
            act,
            exp
        );
        assert!(!act.is_nan(), "NaN in mul");
    }

    // Div
    let c_div = a.broadcast_div(&b)?;
    let v_div = c_div.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    for (act, exp) in v_div.iter().zip([-4.0f32, 2.0, 4.0, 2.0].iter()) {
        assert!(
            (act - exp).abs() < div_tol,
            "div mismatch: {} vs {}",
            act,
            exp
        );
        assert!(!act.is_nan(), "NaN in div");
        assert!(!act.is_infinite(), "Inf in div");
    }

    println!(
        "FP8 binary ops OK: add={:?}, sub={:?}, mul={:?}, div={:?}",
        v_add, v_sub, v_mul, v_div
    );
    Ok(())
}

/// Test sum reduction on F8E4M3
fn sum_fp8(device: &Device) -> Result<()> {
    if device.is_cpu() {
        return Ok(());
    }
    // Create a small tensor of known positive values
    let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();

    let t = Tensor::from_slice(&data, 64, device)?.to_dtype(DType::F8E4M3)?;
    // FP8 has no native sum kernel, so cast to F32 before reducing
    let s = t.to_dtype(DType::F32)?.sum_all()?;
    let result = s.to_vec0::<f32>()?;

    println!("FP8 sum: {}", result);
    assert!(!result.is_nan(), "NaN in FP8 sum");
    assert!(!result.is_infinite(), "Inf in FP8 sum");
    let expected = data.iter().sum::<f32>();
    assert!(
        (result - expected).abs() < 5.0,
        "FP8 sum mismatch: got {result}, expected approximately {expected}"
    );
    Ok(())
}

/// Test FP8 operation chain against an F32 reference.
fn fp8_stability(device: &Device) -> Result<()> {
    if device.is_cpu() {
        return Ok(());
    }
    let a_data = (0..16 * 16)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
        .collect::<Vec<_>>();
    let b_data = (0..16 * 16)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.0625)
        .collect::<Vec<_>>();
    let expected = {
        let a = Tensor::from_vec(a_data.clone(), (16, 16), &Device::Cpu)?;
        let b = Tensor::from_vec(b_data.clone(), (16, 16), &Device::Cpu)?;
        a.broadcast_add(&b)?.broadcast_mul(&b)?
    };
    let a = Tensor::from_vec(a_data, (16, 16), device)?.to_dtype(DType::F8E4M3)?;
    let b = Tensor::from_vec(b_data, (16, 16), device)?.to_dtype(DType::F8E4M3)?;

    let c = a.broadcast_add(&b)?;
    let got = c
        .broadcast_mul(&b)?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?;
    let diff = got.sub(&expected)?.abs()?;
    let max_abs = diff.max_all()?.to_scalar::<f32>()?;
    let mean_abs = diff.mean_all()?.to_scalar::<f32>()?;

    let max_tol = if uses_legacy_fp8(device) { 0.5 } else { 0.35 };
    assert!(
        max_abs.is_finite() && mean_abs.is_finite() && max_abs < max_tol && mean_abs < 0.15,
        "FP8 add/mul chain mismatch against f32 reference: max_abs={max_abs}, mean_abs={mean_abs}"
    );

    Ok(())
}

test_device!(cast_fp8, cast_fp8_cpu, cast_fp8_gpu, cast_fp8_metal);
test_device!(unary_fp8, unary_fp8_cpu, unary_fp8_gpu, unary_fp8_metal);
test_device!(binary_fp8, binary_fp8_cpu, binary_fp8_gpu, binary_fp8_metal);
test_device!(sum_fp8, sum_fp8_cpu, sum_fp8_gpu, sum_fp8_metal);
test_device!(
    fp8_stability,
    fp8_stability_cpu,
    fp8_stability_gpu,
    fp8_stability_metal
);
