//! Hardware correctness tests for the MMQ path.
//!
//! MMQ requantizes the activations to `q8_1`, so like MMVQ it is a numerically
//! *different* computation from the dense dequantize it replaced. Every test
//! below compares it against that same dequantize-then-matmul reference.
//!
//! [`check`] launches the kernel directly rather than going through
//! `QRocmStorage::fwd`, so the shapes it covers are free of the dispatch
//! policy: `mmq::is_faster` declines the smallest weight matrices, and
//! quantizing one past its threshold per test case would cost seconds for no
//! extra coverage of the kernel. The dispatch itself is pinned separately, by
//! [`the_dispatch_reaches_mmq`] and the `is_faster` cases.

use super::mmq;
use crate::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use crate::rocm_backend::{kernels, RocmDevice};
use crate::{DType, Device, Module, Result, Tensor};
use candle_rocm_kernels::MmqTiles;

macro_rules! rocm_device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => Device::Rocm(dev),
            Err(_) => return Ok(()),
        }
    };
}

/// Every dtype with a `mul_mat_q*` kernel.
pub(super) const MMQ_DTYPES: [GgmlDType; 10] = [
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

/// The tolerance every `q8_1` path is held to, relative to the largest
/// magnitude in the reference *row* rather than to each element: a dot product
/// that lands near zero is a cancellation of terms whose own error does not
/// shrink with it. `q8_1` carries seven bits of mantissa and the residual
/// accumulates over `k` terms, which is what sets the 2%.
const TOL: f32 = 2e-2;

fn assert_close(got: &[f32], want: &[f32], row: usize, what: &str) {
    let scale = want.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
    for (col, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (a - b).abs() <= TOL * scale,
            "{what} at ({row}, {col}): mmq {a} vs dequantized {b} (row scale {scale})"
        );
    }
}

/// Launch MMQ for `(m, k) x (n, k)^T` and assert it matches the same product
/// computed from the dequantized weights.
fn check(device: &Device, dtype: GgmlDType, m: usize, k: usize, n: usize) -> Result<()> {
    let dev = match device {
        Device::Rocm(dev) => dev.clone(),
        _ => return Ok(()),
    };
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
    // `QTensor` stores the weights transposed, i.e. `(n, k)`.
    let rhs = Tensor::from_slice(&rhs, (n, k), device)?;
    let qt = QTensor::quantize(&rhs, dtype)?;

    let want = Tensor::from_slice(&lhs, (m, k), device)?
        .matmul(&qt.dequantize(device)?.t()?)?
        .to_vec2::<f32>()?;

    let q = match &qt.storage {
        QStorage::Rocm(q) => q,
        _ => return Ok(()),
    };
    let y = dev.clone_htod(&lhs)?;
    let dst = mmq::mul_mat_via_q8_1(&q.data, q.len, &y, 0, dtype, k, n, m, &dev)?;
    let got = dev.clone_dtoh(&dst)?;

    assert_eq!(got.len(), m * n, "{dtype:?} m={m} k={k} n={n}");
    for row in 0..m {
        assert_close(
            &got[row * n..(row + 1) * n],
            &want[row],
            row,
            &format!("{dtype:?} m={m} k={k} n={n}"),
        );
    }
    Ok(())
}

/// The prefill shape, for every dtype that has a kernel. `k = 512` is the
/// coarsest `k_step` (Q2K, Q3K), so one value covers every dtype.
#[test]
fn mmq_matches_a_dequantized_matmul() -> Result<()> {
    let device = rocm_device!();
    for dtype in MMQ_DTYPES {
        for (m, n) in [(9usize, 64usize), (64, 256)] {
            check(&device, dtype, m, 512, n)?;
        }
    }
    Ok(())
}

/// `mmq_x` and `mmq_y` are 64 or 128 depending on the dtype, so shapes that do
/// not divide by them are what exercise the kernel's own bound checks — the
/// last workgroup in each dimension overhangs the output. Getting this wrong is
/// what MMVQ does at an odd row count, and there it corrupts silently.
#[test]
fn mmq_handles_partial_tiles() -> Result<()> {
    let device = rocm_device!();
    for dtype in [GgmlDType::Q4_0, GgmlDType::Q8_0, GgmlDType::Q6K] {
        for m in [1usize, 9, 65, 130] {
            for n in [1usize, 3, 65, 129] {
                check(&device, dtype, m, 512, n)?;
            }
        }
    }
    Ok(())
}

/// More than one step of the kernel's column loop, and a `k` past the 512-wide
/// row padding the `q8_1` activations carry.
#[test]
fn mmq_handles_a_long_shared_dimension() -> Result<()> {
    let device = rocm_device!();
    for dtype in [GgmlDType::Q4_0, GgmlDType::Q4K] {
        for k in [1024usize, 2048, 4096] {
            check(&device, dtype, 32, k, 64)?;
        }
    }
    Ok(())
}

/// The macro token `quantized.cu` spells `dtype` with, e.g. `Q2K` -> `Q2_K`.
fn macro_dtype(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::Q4_0 => "Q4_0",
        GgmlDType::Q4_1 => "Q4_1",
        GgmlDType::Q5_0 => "Q5_0",
        GgmlDType::Q5_1 => "Q5_1",
        GgmlDType::Q8_0 => "Q8_0",
        GgmlDType::Q2K => "Q2_K",
        GgmlDType::Q3K => "Q3_K",
        GgmlDType::Q4K => "Q4_K",
        GgmlDType::Q5K => "Q5_K",
        GgmlDType::Q6K => "Q6_K",
        other => panic!("no mul_mat_q kernel for {other:?}"),
    }
}

/// The value of `#define <name> <n>` in the kernel source.
///
/// The *last* definition wins, which is not pedantry: the `MMQ_*_AMPERE` names
/// are defined twice, once under `#if defined(CUDA_USE_TENSOR_CORES)` and once
/// under its `#else`. Nothing defines that macro, so the second is the live one.
fn kernel_define(name: &str) -> usize {
    let value = kernels::QUANTIZED
        .source()
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            if fields.next() != Some("#define") || fields.next() != Some(name) {
                return None;
            }
            fields.next()
        })
        .next_back()
        .unwrap_or_else(|| panic!("no #define {name} in quantized.cu"));
    value
        .parse()
        .unwrap_or_else(|e| panic!("#define {name} {value}: {e}"))
}

/// The host computes the grid and block from its own copy of the tile geometry,
/// so a table that disagrees with the kernel's `#define`s is a wrong launch —
/// a workgroup count that leaves part of the output untouched, or a block whose
/// `threadIdx.y` runs past the `nwarps` the kernel indexes with. This checks the
/// two against each other for every dtype and every architecture, which needs no
/// GPU: the kernel source is compiled into the binary.
#[test]
fn the_host_tile_geometry_matches_the_kernel_source() {
    for (tiles, set) in [(MmqTiles::Ampere, "AMPERE"), (MmqTiles::Rdna2, "RDNA2")] {
        for dtype in MMQ_DTYPES {
            let d = macro_dtype(dtype);
            let want = (
                kernel_define(&format!("MMQ_X_{d}_{set}")),
                kernel_define(&format!("MMQ_Y_{d}_{set}")),
                kernel_define(&format!("NWARPS_{d}_{set}")),
            );
            assert_eq!(mmq::geometry(dtype, tiles), Some(want), "{dtype:?} {set}");
        }
    }
}

/// The `k` and dtype conditions are correctness ones: a `k` the column loop
/// cannot step over evenly makes the last step read into the *next* weight row,
/// which is a wrong answer rather than a fault.
#[test]
fn unsupported_shapes_are_declined() {
    assert!(mmq::supports(GgmlDType::Q4_0, 256));
    assert!(!mmq::supports(GgmlDType::Q4_0, 128));
    // Q2K and Q3K step 512 elements at a time, twice the others.
    assert!(mmq::supports(GgmlDType::Q2K, 512));
    assert!(!mmq::supports(GgmlDType::Q2K, 256));
    assert!(!mmq::supports(GgmlDType::Q4_0, 0));
    // No `mul_mat_q*` entry point exists for these at all.
    for dtype in [GgmlDType::Q8_1, GgmlDType::F16, GgmlDType::F32] {
        assert!(!mmq::supports(dtype, 512));
    }
}

/// Smallest and largest weight counts `bench_prefill_paths` sweeps a crossover
/// across: 512x512, and the 2048x2048 where the last dtype (Q2K) catches up.
/// Everything `mmq::min_work` claims has to sit between them — below the floor
/// there is no measurement to justify enabling MMQ, and above the ceiling the
/// sweep says MMQ wins for every dtype, so declining it wastes the measurement.
const SWEEP_FLOOR: usize = 256 << 10;
const LAST_MEASURED_CROSSOVER: usize = 4 << 20;

/// The weight count at which `mmq::is_faster` switches on, or `None` if it never
/// does. Found by bisection, which is also what pins the policy's *shape*: a
/// single boundary with nothing enabled below it and everything above.
fn threshold(dtype: GgmlDType) -> Option<usize> {
    let ceiling = 1 << 30;
    if !mmq::is_faster(dtype, ceiling) {
        return None;
    }
    let (mut lo, mut hi) = (0usize, ceiling);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if mmq::is_faster(dtype, mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Some(lo)
}

/// The policy half of the dispatch: MMQ is taken on the strength of the weight
/// matrix alone. This checks the properties the measurement establishes rather
/// than the constants it produced, so retuning `mmq::min_work` against a fresh
/// sweep does not mean rewriting the test to agree with it. See `min_work` for
/// the table.
#[test]
fn the_size_policy_matches_what_was_measured() {
    for dtype in MMQ_DTYPES {
        // Under the RDNA geometry every dtype with a kernel wins somewhere, so
        // an MMQ dtype that never dispatches is a policy that threw away a
        // measured win.
        let Some(min) = threshold(dtype) else {
            panic!("{dtype:?} has a mul_mat_q kernel but takes MMQ at no size")
        };
        assert!(
            min >= SWEEP_FLOOR,
            "{dtype:?} enabled at {min} weights, below the {SWEEP_FLOOR} the sweep bottoms out at"
        );
        assert!(
            min <= LAST_MEASURED_CROSSOVER,
            "{dtype:?} declines MMQ up to {min} weights, past where the sweep has it winning"
        );
        assert!(!mmq::is_faster(dtype, min - 1), "{dtype:?} below {min}");
        assert!(mmq::is_faster(dtype, min), "{dtype:?} at {min}");
    }

    // No `mul_mat_q*` entry point, so no threshold could make this dispatch.
    for dtype in [GgmlDType::Q8_1, GgmlDType::F16, GgmlDType::F32] {
        assert_eq!(threshold(dtype), None, "{dtype:?}");
    }

    // The measured ordering. Seven dtypes beat dense at every size swept and so
    // share the floor; Q6K, Q3K and Q2K each crossed over inside the sweep, in
    // that order, and each therefore has to wait strictly longer than the last.
    for dtype in [
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
    ] {
        assert_eq!(threshold(dtype), Some(SWEEP_FLOOR), "{dtype:?}");
    }
    let mut prev = SWEEP_FLOOR;
    for dtype in [GgmlDType::Q6K, GgmlDType::Q3K, GgmlDType::Q2K] {
        let min = threshold(dtype).unwrap_or_default();
        assert!(
            min > prev,
            "{dtype:?} crosses over at {min}, not after {prev}"
        );
        prev = min;
    }
    assert_eq!(prev, LAST_MEASURED_CROSSOVER);
}

/// End to end through `QRocmStorage::fwd`, at a shape the policy actually sends
/// to MMQ: 2048x2048 Q4_0 is 4 Mi weights, well past its threshold. f16 and
/// bf16 activations reach the kernel through a cast to f32, which is the other
/// half of `try_fwd`.
#[test]
fn the_dispatch_reaches_mmq() -> Result<()> {
    let device = rocm_device!();
    let (m, k, n) = (16usize, 2048usize, 2048usize);
    assert!(mmq::is_faster(GgmlDType::Q4_0, n * k));

    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
    let rhs = Tensor::from_slice(&rhs, (n, k), &device)?;
    let qt = QTensor::quantize(&rhs, GgmlDType::Q4_0)?;
    let deq = qt.dequantize(&device)?.t()?;
    let mm = QMatMul::from_qtensor(qt)?;

    for act in [DType::F32, DType::F16, DType::BF16] {
        let lhs = Tensor::from_slice(&lhs, (m, k), &device)?.to_dtype(act)?;
        let want = lhs.to_dtype(DType::F32)?.matmul(&deq)?.to_vec2::<f32>()?;
        let got = mm.forward(&lhs)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        for (row, (got, want)) in got.iter().zip(want.iter()).enumerate() {
            assert_close(got, want, row, &format!("dispatch {act:?}"));
        }
    }
    Ok(())
}

/// The `(b, m, k)` spelling flattens to the same batch as `(b * m, k)`, which
/// is the shape a transformer actually hands the matmul at prefill.
#[test]
fn three_dimensional_prefill_activations_match() -> Result<()> {
    let device = rocm_device!();
    let (b, m, k, n) = (2usize, 24usize, 2048usize, 2048usize);
    let lhs: Vec<f32> = (0..b * m * k).map(|i| (i as f32 / 53.).sin()).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
    let lhs = Tensor::from_slice(&lhs, (b, m, k), &device)?;
    let rhs = Tensor::from_slice(&rhs, (n, k), &device)?;

    let qt = QTensor::quantize(&rhs, GgmlDType::Q4_0)?;
    let want = lhs.matmul(
        &qt.dequantize(&device)?
            .t()?
            .broadcast_as((b, k, n))?
            .contiguous()?,
    )?;
    let got = QMatMul::from_qtensor(qt)?.forward(&lhs)?;
    assert_eq!(got.dims(), &[b, m, n]);

    let got = got.flatten_all()?.to_vec1::<f32>()?;
    let want = want.flatten_all()?.to_vec1::<f32>()?;
    assert_close(&got, &want, 0, "3-D prefill");
    Ok(())
}
