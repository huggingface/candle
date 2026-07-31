//! Hardware correctness tests for the MMVQ path.
//!
//! MMVQ requantizes the activations to `q8_1` before the dot product, so it is
//! a numerically *different* computation from both DMMV and the dense
//! fallback — and it is the path every decode step now takes. Each test below
//! compares it against the same dequantize-then-matmul reference
//! `dmmv_decode_matches_a_dequantized_matmul` uses.

use crate::quantized::{GgmlDType, QMatMul, QTensor};
use crate::rocm_backend::RocmDevice;
use crate::{DType, Device, Module, Result, Tensor};

/// `RocmDevice::new` fails on machines without a GPU; those runs skip.
macro_rules! rocm_device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => Device::Rocm(dev),
            Err(_) => return Ok(()),
        }
    };
}

/// Every dtype with an MMVQ kernel.
const MMVQ_DTYPES: [GgmlDType; 10] = [
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
];

/// Run `(m, k) x (n, k)^T` both quantized and dequantized, and assert they
/// agree.
///
/// The tolerance is relative to the largest magnitude in the *reference* row,
/// not to each element: a dot product that happens to land near zero is a
/// cancellation of terms whose own error does not shrink with it, so a
/// per-element relative bound would be meaningless there. `q8_1` carries seven
/// bits of mantissa, and the residual accumulates over `k` terms, which is
/// what sets the 2% figure.
fn check(
    device: &Device,
    dtype: GgmlDType,
    m: usize,
    k: usize,
    n: usize,
    act: DType,
) -> Result<()> {
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
    let lhs = Tensor::from_slice(&lhs, (m, k), device)?.to_dtype(act)?;
    // `QTensor` stores the weights transposed, i.e. `(n, k)`.
    let rhs = Tensor::from_slice(&rhs, (n, k), device)?;

    let qt = QTensor::quantize(&rhs, dtype)?;
    let want = lhs
        .to_dtype(DType::F32)?
        .matmul(&qt.dequantize(device)?.t()?)?
        .to_vec2::<f32>()?;
    let got = QMatMul::from_qtensor(qt)?
        .forward(&lhs)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;

    assert_eq!(got.len(), m, "{dtype:?} m={m} k={k} n={n} {act:?}");
    for (row, (got, want)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(got.len(), n, "{dtype:?} m={m} k={k} n={n} {act:?}");
        let scale = want.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
        for (col, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 2e-2 * scale,
                "{dtype:?} m={m} k={k} n={n} {act:?} at ({row}, {col}): \
                 mmvq {a} vs dequantized {b} (row scale {scale})"
            );
        }
    }
    Ok(())
}

/// The decode shape, for every dtype that has a kernel. `k = 4096` runs more
/// than one iteration of the kernel's column loop and `n = 128` spans several
/// workgroups.
#[test]
fn mmvq_decode_matches_a_dequantized_matmul() -> Result<()> {
    let device = rocm_device!();
    for dtype in MMVQ_DTYPES {
        for (k, n) in [(256usize, 4usize), (4096, 128)] {
            check(&device, dtype, 1, k, n, DType::F32)?;
        }
    }
    Ok(())
}

/// Batches 2..=8 select different kernel instantiations *and* a different
/// `rows_per_cuda_block`, so each one is its own launch geometry.
#[test]
fn mmvq_batches_match_a_dequantized_matmul() -> Result<()> {
    let device = rocm_device!();
    for dtype in MMVQ_DTYPES {
        // 4 is the last batch with nwarps = 4, 5 the first with nwarps = 2.
        for m in [2usize, 3, 4, 5, 8] {
            check(&device, dtype, m, 1024, 64, DType::F32)?;
        }
    }
    Ok(())
}

/// f16 and bf16 activations reach MMVQ through a cast to f32; before this path
/// existed they fell all the way through to the dense dequantize.
#[test]
fn mmvq_handles_half_precision_activations() -> Result<()> {
    let device = rocm_device!();
    for dtype in [GgmlDType::Q4_0, GgmlDType::Q4K, GgmlDType::Q6K] {
        for act in [DType::F16, DType::BF16] {
            for m in [1usize, 4, 8] {
                check(&device, dtype, m, 1024, 64, act)?;
            }
        }
    }
    Ok(())
}

/// An odd `n` with a batch is the case MMVQ declines (the kernel would write
/// one row past the output); the result still has to be right, via whichever
/// path picks it up. `n = 1` additionally exercises a grid of a single block.
#[test]
fn odd_output_rows_stay_correct() -> Result<()> {
    let device = rocm_device!();
    for dtype in [GgmlDType::Q4_0, GgmlDType::Q4K] {
        for n in [1usize, 3, 65] {
            for m in [1usize, 2, 8] {
                check(&device, dtype, m, 512, n, DType::F32)?;
            }
        }
    }
    Ok(())
}

/// A batch of 9 is past the largest compiled instantiation and must fall
/// through to the dense path rather than resolve to no symbol.
#[test]
fn batches_past_the_kernel_table_fall_back() -> Result<()> {
    let device = rocm_device!();
    for dtype in [GgmlDType::Q4_0, GgmlDType::Q5K] {
        for m in [9usize, 16, 64] {
            check(&device, dtype, m, 512, 32, DType::F32)?;
        }
    }
    Ok(())
}

/// The `(b, m, k)` spelling flattens to the same batch as `(b * m, k)`, and a
/// 3-D activation is what a transformer actually hands the matmul.
#[test]
fn three_dimensional_activations_match() -> Result<()> {
    let device = rocm_device!();
    let (b, m, k, n) = (2usize, 3usize, 512usize, 64usize);
    for dtype in [GgmlDType::Q4_0, GgmlDType::Q4K] {
        let lhs: Vec<f32> = (0..b * m * k).map(|i| (i as f32 / 53.).sin()).collect();
        let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
        let lhs = Tensor::from_slice(&lhs, (b, m, k), &device)?;
        let rhs = Tensor::from_slice(&rhs, (n, k), &device)?;

        let qt = QTensor::quantize(&rhs, dtype)?;
        let want = lhs.matmul(
            &qt.dequantize(&device)?
                .t()?
                .broadcast_as((b, k, n))?
                .contiguous()?,
        )?;
        let got = QMatMul::from_qtensor(qt)?.forward(&lhs)?;
        assert_eq!(got.dims(), &[b, m, n], "{dtype:?}");

        let got = got.flatten_all()?.to_vec1::<f32>()?;
        let want = want.flatten_all()?.to_vec1::<f32>()?;
        let scale = want.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
        for (i, (a, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - w).abs() <= 2e-2 * scale,
                "{dtype:?} element {i}: mmvq {a} vs dequantized {w}"
            );
        }
    }
    Ok(())
}
