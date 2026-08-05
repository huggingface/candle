use candle_core::{test_device, DType, Device, IndexOp, Result, Tensor};

fn matmul(device: &Device) -> Result<()> {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?;

    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[[7.0f32, 10.0], [15.0, 22.0]]);

    let data = vec![1.0f32, 2.0];
    let a = Tensor::from_slice(&data, (2, 1), device)?;
    let data = vec![3.0f32, 4.0];
    let b = Tensor::from_slice(&data, (1, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[&[3.0, 4.0], &[6.0, 8.0]]);

    let data: Vec<_> = (0..6).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 3), device)?;
    let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (3, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, &[&[16., 19.], &[52., 64.]]);

    let data: Vec<_> = (0..12).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 2, 3), device)?;
    let data: Vec<_> = (0..12).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (2, 3, 2), device)?;
    let expected = [[[16., 19.], [52., 64.]], [[214., 235.], [304., 334.]]];

    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec3::<f32>()?, &expected);

    // Also perform the matmul on contiguous transposed versions.
    let a_tt = a.t()?.contiguous()?.t()?;
    assert!(!a_tt.is_contiguous());
    assert_eq!(a.dims(), a_tt.dims());
    assert_eq!(a_tt.stride(), &[6, 1, 2]);

    let b_tt = b.t()?.contiguous()?.t()?;
    assert!(!b_tt.is_contiguous());
    assert_eq!(b.dims(), b_tt.dims());
    assert_eq!(b_tt.stride(), &[6, 1, 3]);

    assert_eq!(a_tt.matmul(&b)?.to_vec3::<f32>()?, &expected);
    assert_eq!(a.matmul(&b_tt)?.to_vec3::<f32>()?, &expected);
    assert_eq!(a_tt.matmul(&b_tt)?.to_vec3::<f32>()?, &expected);
    Ok(())
}

fn matmul_bf16(device: &Device) -> Result<()> {
    if !device.supports_bf16() {
        return Ok(());
    }
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?.to_dtype(DType::BF16)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?.to_dtype(DType::BF16)?;

    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    assert_eq!(c.to_vec2::<f32>()?, &[[7.0f32, 10.0], [15.0, 22.0]]);
    Ok(())
}

fn broadcast_matmul(device: &Device) -> Result<()> {
    let lhs = Tensor::randn(0f32, 1f32, (3, 1, 4, 5), device)?;
    let rhs = Tensor::randn(0f32, 1f32, (6, 5, 2), device)?;
    let out = lhs.broadcast_matmul(&rhs)?;
    assert_eq!(out.dims(), &[3, 6, 4, 2]);
    for idx1 in 0..3 {
        for idx2 in 0..6 {
            let out = out.i((idx1, idx2))?;
            let lhs = lhs.i((idx1, 0))?;
            let rhs = rhs.i(idx2)?;
            let out2 = lhs.matmul(&rhs);
            let sum_diff2 = (out - out2)?.sqr()?.sum_all()?;
            // With cuda, we see errors of up to ~1e-12.
            assert!(sum_diff2.to_vec0::<f32>()? < 1e-6)
        }
    }
    Ok(())
}

#[test]
fn tensor_dot() -> Result<()> {
    let lhs = Tensor::new(&[1., 2., 3.], &Device::Cpu)?;
    let rhs = Tensor::new(&[4., 5., 6.], &Device::Cpu)?;
    let expected = Tensor::new(32., &Device::Cpu)?;
    let dot_ret = lhs.dot(&rhs)?;
    candle_core::test_utils::assert_tensor_eq(&dot_ret, &expected)?;
    Ok(())
}

#[test]
fn tensor_mv() -> Result<()> {
    let mat = Tensor::new(&[[1., 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let vec = Tensor::new(&[1., 1., 1.], &Device::Cpu)?;
    let expected = Tensor::new(&[6., 15.], &Device::Cpu)?;
    let mv_ret = mat.mv(&vec)?;
    candle_core::test_utils::assert_tensor_eq(&mv_ret, &expected)?;
    Ok(())
}

// https://github.com/huggingface/candle/issues/1948
fn squeeze_mm(device: &Device) -> Result<()> {
    let seq_len = 8_usize;
    let a = Tensor::zeros((1, seq_len, 16), DType::F32, device)?;
    let x = a.i((.., seq_len - 1, ..))?;
    let w = Tensor::zeros((32, 16), DType::F32, device)?.t()?;
    let x = x.matmul(&w)?;
    assert_eq!(x.dims(), &[1, 32]);
    Ok(())
}

// https://github.com/huggingface/candle/issues/1992
fn mm_layout(device: &Device) -> Result<()> {
    let a = Tensor::arange(0f32, 16f32, device)?.reshape((1, 1, 4, 4))?;
    let b = Tensor::arange(0f32, 8f32, device)?.reshape((1, 1, 4, 2))?;
    let mm1 = a.matmul(&b)?;
    // Forces the layout to be:
    // shape: [1, 1, 4, 2], stride: [8, 2, 2, 1], start_offset: 0
    // This is still a contiguous matrix but matmul checks are only the two last dimensions have
    // non 1 sizes but matmul check may be reluctant to handle it.
    let b = b.transpose(1, 2)?.force_contiguous()?.transpose(1, 2)?;
    let mm2 = a.matmul(&b)?;
    let diff = (mm1 - mm2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    Ok(())
}

/// FP8 E4M3 matmul with NN layout (both contiguous).
/// Falls back to cast→BF16 since FP8 tensor cores require TN layout.
fn matmul_fp8(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    let a_data: Vec<f32> = (0..256).map(|i| (i % 7) as f32 * 0.5 - 1.5).collect();
    let b_data: Vec<f32> = (0..256).map(|i| (i % 5) as f32 * 0.4 - 0.8).collect();
    let a_f32 = Tensor::from_slice(&a_data, (16, 16), device)?;
    let b_f32 = Tensor::from_slice(&b_data, (16, 16), device)?;
    let expected = a_f32.matmul(&b_f32)?;

    let a_fp8 = a_f32.to_dtype(DType::F8E4M3)?;
    let b_fp8 = b_f32.to_dtype(DType::F8E4M3)?;
    let c = a_fp8.matmul(&b_fp8)?;
    // NN layout falls back to BF16 matmul
    assert_eq!(c.dtype(), DType::BF16);

    let c_f32 = c.to_dtype(DType::F32)?;
    let diff = (&expected - &c_f32)?.abs()?.max(0)?.max(0)?;
    let max_err = diff.to_vec0::<f32>()?;
    assert!(max_err < 2.0, "fp8 matmul max error: {max_err}");
    Ok(())
}

/// FP8 matmul with TN layout (A transposed, B contiguous).
/// This exercises native FP8 tensor cores via cuBLASLt.
/// Matches nn::Linear pattern: x.matmul(&w.t())
fn matmul_fp8_rect(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    // Simulate x (32, 64) @ w.t() where w is (48, 64) → w.t() is (64, 48)
    let m = 32;
    let k = 64;
    let n = 48;
    let x_data: Vec<f32> = (0..(m * k)).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let w_data: Vec<f32> = (0..(n * k)).map(|i| (i % 9) as f32 * 0.1 - 0.4).collect();
    let x_f32 = Tensor::from_slice(&x_data, (m, k), device)?;
    let w_f32 = Tensor::from_slice(&w_data, (n, k), device)?;
    let expected = x_f32.matmul(&w_f32.t()?)?; // x @ w.t()

    let x_fp8 = x_f32.to_dtype(DType::F8E4M3)?;
    let w_fp8 = w_f32.to_dtype(DType::F8E4M3)?;
    let c = x_fp8.matmul(&w_fp8.t()?)?; // TN layout → native FP8 matmul
    assert_eq!(c.dtype(), DType::BF16);

    let c_f32 = c.to_dtype(DType::F32)?;
    let diff = (&expected - &c_f32)?.abs()?.max(0)?.max(0)?;
    let max_err = diff.to_vec0::<f32>()?;
    assert!(max_err < 2.0, "fp8 TN matmul max error: {max_err}");
    Ok(())
}

/// FP8 batched matmul with TN layout.
fn matmul_fp8_batched(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    // Batch of 2: x (2, 16, 64) @ w.t() where w is (2, 32, 64) → w.t() is (2, 64, 32)
    let batch = 2;
    let m = 16;
    let k = 64;
    let n = 32;
    let x_data: Vec<f32> = (0..(batch * m * k))
        .map(|i| (i % 13) as f32 * 0.1 - 0.6)
        .collect();
    let w_data: Vec<f32> = (0..(batch * n * k))
        .map(|i| (i % 7) as f32 * 0.1 - 0.3)
        .collect();
    let x_f32 = Tensor::from_slice(&x_data, (batch, m, k), device)?;
    let w_f32 = Tensor::from_slice(&w_data, (batch, n, k), device)?;
    let expected = x_f32.matmul(&w_f32.transpose(1, 2)?)?;

    let x_fp8 = x_f32.to_dtype(DType::F8E4M3)?;
    let w_fp8 = w_f32.to_dtype(DType::F8E4M3)?;
    let c = x_fp8.matmul(&w_fp8.transpose(1, 2)?)?;
    assert_eq!(c.dtype(), DType::BF16);
    assert_eq!(c.dims(), &[batch, m, n]);

    let c_f32 = c.to_dtype(DType::F32)?;
    let diff = (&expected - &c_f32)?.abs()?;
    let max_err = diff.max(0)?.max(0)?.max(0)?.to_vec0::<f32>()?;
    assert!(max_err < 4.0, "fp8 batched TN matmul max error: {max_err}");
    Ok(())
}

/// Mixed BF16 × FP8 matmul — simulates FP8 weights with BF16 activations.
/// This is the "manual cast" pattern used by ComfyUI for FP8 inference.
fn matmul_fp8_mixed(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    // Simulate nn::Linear: x (BF16) @ w.t() (FP8)
    let m = 32;
    let k = 64;
    let n = 48;
    let x_data: Vec<f32> = (0..(m * k)).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
    let w_data: Vec<f32> = (0..(n * k)).map(|i| (i % 9) as f32 * 0.1 - 0.4).collect();
    let x_bf16 = Tensor::from_slice(&x_data, (m, k), device)?.to_dtype(DType::BF16)?;
    let w_f32 = Tensor::from_slice(&w_data, (n, k), device)?;
    let expected = x_bf16.to_dtype(DType::F32)?.matmul(&w_f32.t()?)?;

    // FP8 weight with BF16 activation — should auto-cast FP8→BF16
    let w_fp8 = w_f32.to_dtype(DType::F8E4M3)?;
    let c = x_bf16.matmul(&w_fp8.t()?)?;
    assert_eq!(c.dtype(), DType::BF16);

    let c_f32 = c.to_dtype(DType::F32)?;
    let diff = (&expected - &c_f32)?.abs()?.max(0)?.max(0)?;
    let max_err = diff.to_vec0::<f32>()?;
    assert!(max_err < 2.0, "mixed BF16×FP8 matmul max error: {max_err}");
    Ok(())
}

test_device!(
    matmul_fp8_mixed,
    matmul_fp8_mixed_cpu,
    matmul_fp8_mixed_gpu,
    matmul_fp8_mixed_metal
);
test_device!(matmul, matmul_cpu, matmul_gpu, matmul_metal);
test_device!(
    matmul_bf16,
    matmul_bf16_cpu,
    matmul_bf16_gpu,
    matmul_bf16_metal
);
test_device!(
    broadcast_matmul,
    broadcast_matmul_cpu,
    broadcast_matmul_gpu,
    broadcast_matmul_metal
);
test_device!(squeeze_mm, squeeze_mm_cpu, squeeze_mm_gpu, squeeze_mm_metal);
test_device!(mm_layout, mm_layout_cpu, mm_layout_gpu, mm_layout_metal);
test_device!(matmul_fp8, matmul_fp8_cpu, matmul_fp8_gpu, matmul_fp8_metal);
test_device!(
    matmul_fp8_rect,
    matmul_fp8_rect_cpu,
    matmul_fp8_rect_gpu,
    matmul_fp8_rect_metal
);
test_device!(
    matmul_fp8_batched,
    matmul_fp8_batched_cpu,
    matmul_fp8_batched_gpu,
    matmul_fp8_batched_metal
);
