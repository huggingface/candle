//! The dense fallback, and the dispatch that now prefers it at prefill.
//!
//! The reorientation is a layout change, so a mistake in it lands scrambled
//! values rather than drifted ones — hence a plain f32 matmul as the reference.
//! The dispatch is a policy change, so `preferred` is pinned on both sides of
//! its threshold and on the cases it must refuse.

use super::dense;
use crate::quantized::{GgmlDType, QMatMul, QTensor};
use crate::rocm_backend::RocmDevice;
use crate::{DType, Device, Module, Result, Tensor};

macro_rules! rocm_device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => Device::Rocm(dev),
            Err(_) => return Ok(()),
        }
    };
}

/// rocBLAS accumulates f16 in f32, but the operands carry ten bits of mantissa
/// and the sum runs over `k` terms. Relative to the row's largest magnitude,
/// for the reason `tests_mmq` gives.
const TOL: f32 = 1e-2;

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    let scale = want.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (a - b).abs() <= TOL * scale,
            "{what}: element {i} was {a}, expected {b} (scale {scale})"
        );
    }
}

#[test]
fn preferred_only_where_the_sweep_says_so() {
    let (n, k) = (2048usize, 2048usize);
    // The threshold is on the batch, and 256 is the first that qualifies.
    assert!(!dense::preferred(DType::F16, GgmlDType::Q8_0, n, k, 255));
    assert!(dense::preferred(DType::F16, GgmlDType::Q8_0, n, k, 256));
    assert!(dense::preferred(DType::F16, GgmlDType::Q8_0, n, k, 4096));

    // f32 and bf16 keep MMQ at every batch: neither was measured ahead of it,
    // and bf16 has no dequantize kernel to reach this path cheaply.
    assert!(!dense::preferred(DType::F32, GgmlDType::Q8_0, n, k, 4096));
    assert!(!dense::preferred(DType::BF16, GgmlDType::Q8_0, n, k, 4096));

    // A vocabulary-sized matrix would hold half a gigabyte of transient.
    assert!(!dense::preferred(
        DType::F16,
        GgmlDType::Q8_0,
        32000,
        4096,
        4096
    ));
    // ...while every projection in a model this card runs is under the cap.
    assert!(dense::preferred(
        DType::F16,
        GgmlDType::Q8_0,
        8192,
        2048,
        1024
    ));

    // A dtype with no dequantize kernel would have to round-trip through the
    // host to reach this path, so it stays with MMQ and the fallback.
    assert!(!dense::preferred(DType::F16, GgmlDType::Q8_1, n, k, 4096));
}

/// The reoriented path against a plain f32 matmul of the same dequantized
/// weights — the test that the reorientation is the identity on the result.
#[test]
fn the_reoriented_dense_path_matches_a_dequantized_matmul() -> Result<()> {
    let device = rocm_device!();
    // Past `dense::MIN_BATCH`, so `fwd` takes this path rather than MMQ.
    let m = 256usize;
    // Non-square, so a transposed result cannot pass by coincidence.
    for (k, n) in [(2048usize, 512usize), (512, 2048)] {
        let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
        let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
        let lhs = Tensor::from_slice(&lhs, (m, k), &device)?;
        let rhs = Tensor::from_slice(&rhs, (n, k), &device)?;

        for dtype in [GgmlDType::Q8_0, GgmlDType::Q4K, GgmlDType::Q4_0] {
            let qt = QTensor::quantize(&rhs, dtype)?;
            let want = lhs
                .matmul(&qt.dequantize(&device)?.t()?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let got = QMatMul::from_qtensor(qt)?
                .forward(&lhs.to_dtype(DType::F16)?)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_close(&got, &want, &format!("{dtype:?} {k}x{n}"));
        }
    }
    Ok(())
}

/// A transformer hands the matmul `(b, m, k)` with the rhs broadcast over `b`,
/// which is where a zero batch stride reaches rocBLAS.
#[test]
fn the_reoriented_dense_path_handles_a_batched_activation() -> Result<()> {
    let device = rocm_device!();
    let (b, m, k, n) = (2usize, 192usize, 2048usize, 1024usize);
    let lhs: Vec<f32> = (0..b * m * k).map(|i| (i as f32 / 53.).sin()).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
    let lhs = Tensor::from_slice(&lhs, (b, m, k), &device)?;
    let rhs = Tensor::from_slice(&rhs, (n, k), &device)?;
    // b * m = 384, past the threshold, though neither factor is on its own.
    assert!(dense::preferred(DType::F16, GgmlDType::Q8_0, n, k, b * m));

    let qt = QTensor::quantize(&rhs, GgmlDType::Q8_0)?;
    let want = lhs
        .matmul(&qt.dequantize(&device)?.t()?.broadcast_as((b, k, n))?)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let got = QMatMul::from_qtensor(qt)?.forward(&lhs.to_dtype(DType::F16)?)?;
    assert_eq!(got.dims(), &[b, m, n]);
    let got = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_close(&got, &want, "batched");
    Ok(())
}

/// Below the threshold the answer still has to come out of MMQ unchanged —
/// the case where a mistake in `preferred` silently changes which kernel ran.
#[test]
fn the_paths_below_the_threshold_still_agree() -> Result<()> {
    let device = rocm_device!();
    let (k, n) = (2048usize, 1024usize);
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
    let rhs = Tensor::from_slice(&rhs, (n, k), &device)?;
    let qt = QTensor::quantize(&rhs, GgmlDType::Q8_0)?;
    let deq = qt.dequantize(&device)?.t()?;
    let mm = QMatMul::from_qtensor(qt)?;

    for m in [1usize, 64, 255, 256] {
        let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
        let lhs = Tensor::from_slice(&lhs, (m, k), &device)?;
        let want = lhs.matmul(&deq)?.flatten_all()?.to_vec1::<f32>()?;
        let got = mm
            .forward(&lhs.to_dtype(DType::F16)?)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // MMQ requantizes the activations to q8_1, which is a coarser
        // computation than the f16 GEMM above it; hold it to `tests_mmq`'s 2%.
        let scale = want.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 2e-2 * scale,
                "m={m}: element {i} was {a}, expected {b}"
            );
        }
    }
    Ok(())
}
