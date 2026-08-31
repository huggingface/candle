//! ROCm-only dtype coverage for the fused `softmax` / `rms_norm` / `layer_norm`
//! launchers.
//!
//! `tests/ops.rs` is shared with the CPU, CUDA and Metal backends and only ever
//! exercises f32, which is how bf16 — the dtype most `candle-transformers`
//! models run in — stayed missing from all three ROCm dispatches while working
//! on CUDA. Everything here is checked against the CPU reference.

use candle::{DType, Device, Result, Tensor};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
    let lhs = lhs.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let rhs = rhs.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    (lhs - rhs)?.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()
}

/// bf16 tolerances are loose on purpose: the kernels accumulate in f32 while the
/// CPU reference works in the storage dtype, so the two round differently.
const DTYPES: [(DType, f32); 4] = [
    (DType::BF16, 5e-2),
    (DType::F16, 5e-3),
    (DType::F32, 1e-5),
    (DType::F64, 1e-5),
];

#[test]
fn softmax_last_dim_dtype_coverage() -> Result<()> {
    let dev = Device::new_rocm(0)?;
    let cpu = Device::Cpu;
    let src = Tensor::rand(-2f32, 2f32, (3, 7, 24), &cpu)?;
    for (dtype, tol) in DTYPES {
        let src = src.to_dtype(dtype)?;
        let gpu = super::softmax_last_dim(&src.to_device(&dev)?)?;
        assert_eq!(gpu.dtype(), dtype);
        let want = super::softmax_last_dim(&src)?;
        let diff = max_abs_diff(&gpu, &want)?;
        assert!(
            diff <= tol,
            "softmax {dtype:?}: max abs diff {diff} > {tol}"
        );
    }
    Ok(())
}

#[test]
fn rms_norm_dtype_coverage() -> Result<()> {
    let dev = Device::new_rocm(0)?;
    let cpu = Device::Cpu;
    let src = Tensor::rand(-2f32, 2f32, (3, 7, 24), &cpu)?;
    let alpha = Tensor::rand(0f32, 1f32, 24, &cpu)?;
    for (dtype, tol) in DTYPES {
        let (src, alpha) = (src.to_dtype(dtype)?, alpha.to_dtype(dtype)?);
        let gpu = super::rms_norm(&src.to_device(&dev)?, &alpha.to_device(&dev)?, 1e-5)?;
        assert_eq!(gpu.dtype(), dtype);
        let want = super::rms_norm_slow(&src, &alpha, 1e-5)?;
        let diff = max_abs_diff(&gpu, &want)?;
        assert!(
            diff <= tol,
            "rms_norm {dtype:?}: max abs diff {diff} > {tol}"
        );
    }
    Ok(())
}

#[test]
fn layer_norm_dtype_coverage() -> Result<()> {
    let dev = Device::new_rocm(0)?;
    let cpu = Device::Cpu;
    let src = Tensor::rand(-2f32, 2f32, (3, 7, 24), &cpu)?;
    let alpha = Tensor::rand(0f32, 1f32, 24, &cpu)?;
    let beta = Tensor::rand(-1f32, 1f32, 24, &cpu)?;
    for (dtype, tol) in DTYPES {
        let (src, alpha, beta) = (
            src.to_dtype(dtype)?,
            alpha.to_dtype(dtype)?,
            beta.to_dtype(dtype)?,
        );
        let gpu = super::layer_norm(
            &src.to_device(&dev)?,
            &alpha.to_device(&dev)?,
            &beta.to_device(&dev)?,
            1e-5,
        )?;
        assert_eq!(gpu.dtype(), dtype);
        let want = super::layer_norm_slow(&src, &alpha, &beta, 1e-5)?;
        let diff = max_abs_diff(&gpu, &want)?;
        assert!(
            diff <= tol,
            "layer_norm {dtype:?}: max abs diff {diff} > {tol}"
        );
    }
    Ok(())
}
