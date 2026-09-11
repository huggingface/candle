//! Imatrix quantization runs on the host for both backends, so the ROCm result
//! has to be byte-identical to the CPU one — not merely close.

use super::*;
use crate::quantized::QTensor;
use crate::{Device, Tensor};

macro_rules! rocm_device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => dev,
            Err(_) => return Ok(()),
        }
    };
}

const ROWS: usize = 4;
const N_PER_ROW: usize = 256;

/// The K-quants the CPU has a `from_float_imatrix` for.
const IMATRIX_DTYPES: [GgmlDType; 5] = [
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
];

fn weights() -> Vec<f32> {
    (0..ROWS * N_PER_ROW)
        .map(|i| (i as f32 * 0.37).sin() * (1. + (i % 13) as f32 / 8.))
        .collect()
}

/// Deliberately far from flat: a uniform imatrix would quantize to the same
/// bytes as the plain quantizer for some dtypes and hide a mis-wired call.
fn imatrix() -> Vec<f32> {
    (0..N_PER_ROW)
        .map(|j| 0.05 + ((j % 17) as f32) * 0.5)
        .collect()
}

#[test]
fn imatrix_matches_the_cpu_backend() -> Result<()> {
    let dev = rocm_device!();
    let device = Device::Rocm(dev);
    let (xs, imatrix) = (weights(), imatrix());
    for dtype in IMATRIX_DTYPES {
        let cpu = Tensor::from_vec(xs.clone(), (ROWS, N_PER_ROW), &Device::Cpu)?;
        let expected = QTensor::quantize_imatrix(&cpu, &imatrix, dtype)?;
        let gpu = Tensor::from_vec(xs.clone(), (ROWS, N_PER_ROW), &device)?;
        let got = QTensor::quantize_imatrix(&gpu, &imatrix, dtype)?;
        assert_eq!(
            got.data()?.as_ref(),
            expected.data()?.as_ref(),
            "{dtype:?} imatrix quantization diverged from the cpu"
        );
        // ... and the imatrix actually reached the quantizer.
        let plain = QTensor::quantize(&cpu, dtype)?;
        assert_ne!(
            got.data()?.as_ref(),
            plain.data()?.as_ref(),
            "{dtype:?} imatrix quantization produced the plain quantization"
        );
    }
    Ok(())
}

#[test]
fn imatrix_onto_matches_the_cpu_backend() -> Result<()> {
    let dev = rocm_device!();
    let device = Device::Rocm(dev);
    let (xs, imatrix) = (weights(), imatrix());
    for dtype in IMATRIX_DTYPES {
        let cpu = Tensor::from_vec(xs.clone(), (ROWS, N_PER_ROW), &Device::Cpu)?;
        let expected = QTensor::quantize_imatrix_onto(&cpu, &imatrix, dtype, &Device::Cpu)?;
        let got = QTensor::quantize_imatrix_onto(&cpu, &imatrix, dtype, &device)?;
        assert_eq!(
            got.data()?.as_ref(),
            expected.data()?.as_ref(),
            "{dtype:?} imatrix quantize_onto diverged from the cpu"
        );
    }
    Ok(())
}

/// Dequantizing what the two backends packed must also agree, which catches an
/// upload that lost or misplaced the padding.
#[test]
fn imatrix_round_trips_like_the_cpu_backend() -> Result<()> {
    let dev = rocm_device!();
    let device = Device::Rocm(dev);
    let (xs, imatrix) = (weights(), imatrix());
    let gpu = Tensor::from_vec(xs.clone(), (ROWS, N_PER_ROW), &device)?;
    let cpu = Tensor::from_vec(xs, (ROWS, N_PER_ROW), &Device::Cpu)?;
    for dtype in [GgmlDType::Q4K, GgmlDType::Q6K] {
        let got = QTensor::quantize_imatrix(&gpu, &imatrix, dtype)?.dequantize(&Device::Cpu)?;
        let expected =
            QTensor::quantize_imatrix(&cpu, &imatrix, dtype)?.dequantize(&Device::Cpu)?;
        let diff = (got - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(diff, 0., "{dtype:?} round trip diverged from the cpu");
    }
    Ok(())
}

/// `k_quants` panics for a dtype it has no imatrix quantizer for; the dtype
/// comes from a config file, so the ROCm path has to turn that into an error.
#[test]
fn imatrix_rejects_a_dtype_without_a_cpu_quantizer() -> Result<()> {
    let dev = rocm_device!();
    let device = Device::Rocm(dev);
    let xs = Tensor::from_vec(weights(), (ROWS, N_PER_ROW), &device)?;
    let err = QTensor::quantize_imatrix(&xs, &imatrix(), GgmlDType::Q4_0)
        .expect_err("Q4_0 has no imatrix quantizer");
    assert!(
        err.to_string().contains("imatrix quantization"),
        "unexpected error: {err}"
    );
    Ok(())
}
