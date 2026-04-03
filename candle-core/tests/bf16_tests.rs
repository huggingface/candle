use candle_core::{test_device, DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
fn uses_legacy_bf16(device: &Device) -> bool {
    device.is_cuda() && candle_core::cuda_backend::kernels::capabilities::ALLOW_LEGACY_BF16
}

#[cfg(not(feature = "cuda"))]
fn uses_legacy_bf16(_device: &Device) -> bool {
    false
}

fn unary_bf16(device: &Device) -> Result<()> {
    if device.is_cpu() {
        return Ok(());
    }
    let data = &[-3f32, 1., 4., -0.1, 0.5];
    let a = Tensor::new(data, device)?.to_dtype(DType::BF16)?;

    // Test exp
    let res_bf16 = a.exp()?;
    let res_f32 = res_bf16.to_dtype(DType::F32)?;

    let expected: Vec<f32> = data.iter().map(|&x| x.exp()).collect();
    let actual = res_f32.to_vec1::<f32>()?;
    let exp_tol = if uses_legacy_bf16(device) { 0.2 } else { 0.1 };

    for (e, a) in expected.iter().zip(actual.iter()) {
        let diff = (e - a).abs();
        assert!(
            diff < exp_tol,
            "exp: expected {}, got {} (diff {})",
            e,
            a,
            diff
        );
    }

    // Compare BF16 GELU against an F32 reference on the CPU.
    let res_bf16 = a.gelu()?;
    let actual = res_bf16.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let expected = Tensor::new(data, &Device::Cpu)?.gelu()?.to_vec1::<f32>()?;
    let gelu_tol = if uses_legacy_bf16(device) { 0.05 } else { 0.02 };
    for (e, a) in expected.iter().zip(actual.iter()) {
        let diff = (e - a).abs();
        assert!(
            diff < gelu_tol,
            "gelu: expected {}, got {} (diff {})",
            e,
            a,
            diff
        );
    }

    Ok(())
}

fn sum_bf16(device: &Device) -> Result<()> {
    if device.is_cpu() {
        return Ok(());
    }
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let a = Tensor::new(data.as_slice(), device)?.to_dtype(DType::BF16)?;
    let res = a.sum_all()?;
    let val = res.to_dtype(DType::F32)?.to_vec0::<f32>()?;
    // BF16 precision at ~5000 is approx 32, so error can be significant if summed sequentially
    assert!(
        (val - 4950.0).abs() < 100.0,
        "sum: expected 4950.0, got {}",
        val
    );
    Ok(())
}

fn bf16_random_stability(device: &Device) -> Result<()> {
    if device.is_cpu() {
        return Ok(());
    }
    // Large random tensor
    let a = Tensor::randn(0f32, 1f32, (256, 256), device)?.to_dtype(DType::BF16)?;
    let b = Tensor::randn(0f32, 1f32, (256, 256), device)?.to_dtype(DType::BF16)?;

    // Chain of ops
    let c = a.matmul(&b)?;
    let d = c.exp()?;
    let e = d.log()?; // Should be back to ~c

    let e_f32 = e.to_dtype(DType::F32)?;
    let data = e_f32.to_vec2::<f32>()?;

    for row in data {
        for &val in &row {
            assert!(!val.is_nan(), "NaN detected in bf16 tensor element");
            assert!(!val.is_infinite(), "Inf detected in bf16 tensor element");
        }
    }

    let f = e.sum_all()?;
    let val = f.to_dtype(DType::F32)?.to_vec0::<f32>()?;
    println!("Random stability sum: {}", val);
    Ok(())
}

use rand::{rngs::StdRng, Rng, SeedableRng};

pub fn random_f32_vec(n: usize, seed: u64, low: f32, high: f32) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(low..high)).collect()
}

// let gen_lambda =
// let a = Tensor::randn(0f32, 1f32, (256, 256), device)?.to_dtype(DType::BF16)?;
// let b = a.to_dtype(DType::F32)?;
// let c = b.to_dtype(DType::BF16)?;
// let d = c.to_dtype(DType::F32)?;
// let diff = (a - d)?.abs()?;

/// Test F32 -> BF16 -> F32 cast roundtrip with random values.
fn bf16_cast_roundtrip(device: &Device) -> Result<()> {
    if device.is_cpu() {
        return Ok(());
    }
    // Use range within BF16 limits to avoid f32 overflow and rand panic.
    let bf16_max = half::bf16::MAX.to_f32();
    let gen_lambda = |n, s| random_f32_vec(n, s, -0.4 * bf16_max, 0.4 * bf16_max);

    let a = Tensor::from_vec(gen_lambda(256 * 256, 0), (256, 256), device)?;
    let b = a.to_dtype(DType::BF16)?;
    let c = b.to_dtype(DType::F32)?;
    let d = c.to_dtype(DType::BF16)?;
    let diff = (b - d)?.abs()?;
    let diff_f32 = diff.to_dtype(DType::F32)?;
    let data = diff_f32.to_vec2::<f32>()?;
    for row in data {
        for &val in &row {
            assert!(!val.is_nan(), "NaN detected in bf16 tensor element");
            assert!(!val.is_infinite(), "Inf detected in bf16 tensor element");
        }
    }
    Ok(())
}

fn max_min(device: &Device) -> Result<()> {
    if device.is_cpu() {
        return Ok(());
    }
    let a = Tensor::randn(0f32, 1f32, (256, 256), device)?.to_dtype(DType::BF16)?;
    let b = Tensor::randn(0f32, 1f32, (256, 256), device)?.to_dtype(DType::BF16)?;
    let c = a.broadcast_maximum(&b)?;
    let d = c.to_dtype(DType::F32)?;
    let e = d.to_vec2::<f32>()?;
    for row in e {
        for &val in &row {
            assert!(!val.is_nan(), "NaN detected in bf16 tensor element");
            assert!(!val.is_infinite(), "Inf detected in bf16 tensor element");
        }
    }
    Ok(())
}

fn bf16_mlp_stability(device: &Device) -> Result<()> {
    if device.is_cpu() {
        return Ok(());
    }

    // Create weights and inputs
    let w1 = Tensor::randn(0f32, 0.1f32, (256, 64), device)?.to_dtype(DType::BF16)?;
    let b1 = Tensor::zeros(256, DType::BF16, device)?;

    let w2 = Tensor::randn(0f32, 0.1f32, (64, 256), device)?.to_dtype(DType::BF16)?;
    let b2 = Tensor::zeros(64, DType::BF16, device)?;

    let x = Tensor::randn(0f32, 1f32, (1, 64), device)?.to_dtype(DType::BF16)?;

    // MLP: x -> linear1 -> gelu -> linear2
    let mut h = x.matmul(&w1.t()?)?;
    h = h.broadcast_add(&b1)?;
    h = h.gelu_erf()?;
    h = h.matmul(&w2.t()?)?;
    h = h.broadcast_add(&b2)?;

    let out = h.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    for row in out {
        for val in row {
            assert!(!val.is_nan(), "NaN detected in MLP output: {:?}", val);
            assert!(!val.is_infinite(), "Inf detected in MLP output: {:?}", val);
        }
    }

    println!("MLP stability successful");
    Ok(())
}

fn binary_bf16(device: &Device) -> Result<()> {
    if device.is_cpu() {
        return Ok(());
    }
    let div_tol = if uses_legacy_bf16(device) { 2e-2 } else { 1e-2 };

    // Create tensors
    let a_data = &[1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b_data = &[5.0f32, 4.0, 3.0, 2.0, 1.0];

    let a = Tensor::new(a_data, device)?.to_dtype(DType::BF16)?;
    let b = Tensor::new(b_data, device)?.to_dtype(DType::BF16)?;

    // Add
    let c_add = a.broadcast_add(&b)?;
    let v_add = c_add.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_eq!(v_add, vec![6.0, 6.0, 6.0, 6.0, 6.0]);

    // Sub
    let c_sub = b.broadcast_sub(&a)?;
    let v_sub = c_sub.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_eq!(v_sub, vec![4.0, 2.0, 0.0, -2.0, -4.0]);

    // Mul
    let c_mul = a.broadcast_mul(&b)?;
    let v_mul = c_mul.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_eq!(v_mul, vec![5.0, 8.0, 9.0, 8.0, 5.0]);

    // Div
    let c_div = a.broadcast_div(&b)?;
    let v_div = c_div.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let expected_div = [0.2f32, 0.5, 1.0, 2.0, 5.0];
    for (act, exp) in v_div.iter().zip(expected_div.iter()) {
        assert!(
            (act - exp).abs() < div_tol,
            "div mismatch: act {}, exp {}",
            act,
            exp
        );
    }

    // Max
    let c_max = a.broadcast_maximum(&b)?;
    let v_max = c_max.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_eq!(v_max, vec![5.0, 4.0, 3.0, 4.0, 5.0]);

    // Min
    let c_min = a.broadcast_minimum(&b)?;
    let v_min = c_min.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_eq!(v_min, vec![1.0, 2.0, 3.0, 2.0, 1.0]);

    // Eq
    let c_eq = a.eq(&b)?;
    let v_eq = c_eq.to_vec1::<u8>()?;
    assert_eq!(v_eq, vec![0, 0, 1, 0, 0]);

    Ok(())
}

test_device!(unary_bf16, unary_bf16_cpu, unary_bf16_gpu, unary_bf16_metal);
test_device!(sum_bf16, sum_bf16_cpu, sum_bf16_gpu, sum_bf16_metal);
test_device!(
    binary_bf16,
    binary_bf16_cpu,
    binary_bf16_gpu,
    binary_bf16_metal
);
test_device!(
    bf16_random_stability,
    bf16_random_stability_cpu,
    bf16_random_stability_gpu,
    bf16_random_stability_metal
);
test_device!(max_min, max_min_cpu, max_min_gpu, max_min_metal);
test_device!(
    bf16_cast_roundtrip,
    bf16_cast_roundtrip_cpu,
    bf16_cast_roundtrip_gpu,
    bf16_cast_roundtrip_metal
);

test_device!(
    bf16_mlp_stability,
    bf16_mlp_stability_cpu,
    bf16_mlp_stability_gpu,
    bf16_mlp_stability_metal
);
