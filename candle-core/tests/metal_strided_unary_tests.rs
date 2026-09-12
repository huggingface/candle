#![cfg(feature = "metal")]
//! Regression tests: `recip` and `sign` on non-contiguous (strided) tensors.
//! The strided Metal kernels for these ops exist in candle-metal-kernels but were
//! not wired into the strided unary dispatch, so calling `.recip()` / `.sign()` on
//! a transposed tensor errored with "Metal strided unary ... not implemented"
//! while the CPU backend handled it fine.

use candle_core::{DType, Device, Result, Tensor};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a: Vec<f32> = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let b: Vec<f32> = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max))
}

// Same data as a strided (transposed) tensor on Metal and contiguous on CPU.
fn strided_metal_and_cpu(
    dev: &Device,
    data: &[f32],
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let m = Tensor::from_vec(data.to_vec(), (cols, rows), dev)?
        .to_dtype(dtype)?
        .transpose(0, 1)?;
    assert!(!m.is_contiguous());
    let c = Tensor::from_vec(data.to_vec(), (cols, rows), &Device::Cpu)?
        .to_dtype(dtype)?
        .t()?
        .contiguous()?;
    Ok((m, c))
}

#[test]
fn strided_recip_matches_cpu() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (rows, cols) = (17usize, 40usize);
    // Strictly positive so reciprocals are well-conditioned across dtypes.
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| 0.5 + ((i * 5 + 1) % 61) as f32 / 7.0)
        .collect();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let (m, c) = strided_metal_and_cpu(&dev, &data, rows, cols, dtype)?;
        let got = m.recip()?;
        let want = c.recip()?;
        assert_eq!(got.shape(), want.shape());
        let err = max_abs_diff(&got, &want)?;
        let tol = if dtype == DType::F32 { 1e-5 } else { 5e-3 };
        assert!(err <= tol, "recip {dtype:?}: max_abs_diff={err}");
    }
    Ok(())
}

#[test]
fn strided_sign_matches_cpu() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (rows, cols) = (17usize, 40usize);
    // Mixed signs including zero to exercise all three sign outputs.
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 % 7.0) - 3.0).collect();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let (m, c) = strided_metal_and_cpu(&dev, &data, rows, cols, dtype)?;
        let got = m.sign()?;
        let want = c.sign()?;
        assert_eq!(got.shape(), want.shape());
        let err = max_abs_diff(&got, &want)?;
        assert!(err <= 1e-5, "sign {dtype:?}: max_abs_diff={err}");
    }
    Ok(())
}
