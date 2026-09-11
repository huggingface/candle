//! Timing harness for the quantized matmul paths.
//!
//! Ignored by default — it needs a GPU and takes seconds. Run with:
//!
//! ```text
//! cargo test -p candle-core --features rocm --lib -- \
//!     quantized::rocm::bench --ignored --nocapture --test-threads=1
//! ```
//!
//! `CANDLE_ROCM_FORCE_DMMV=1` in the environment pins the decode shapes to the
//! DMMV kernel, which is how the DMMV and MMVQ paths are compared.
//!
//! [`super::bench_prefill`] holds the prefill-scale sweep and shares the timing
//! helpers below; the filter above picks up both files.

use crate::quantized::{GgmlDType, QMatMul, QTensor};
use crate::{DType, Device, Module, Result, Tensor};
use std::time::Instant;

use crate::rocm_backend::RocmDevice;

macro_rules! rocm_device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => Device::Rocm(dev),
            Err(_) => return Ok(()),
        }
    };
}

/// Mean wall time of one `QMatMul::forward` over `iters` runs, in milliseconds.
fn time_matmul(
    device: &Device,
    dtype: GgmlDType,
    m: usize,
    k: usize,
    n: usize,
    act: DType,
    iters: usize,
) -> Result<f64> {
    let rhs = Tensor::rand(-1f32, 1f32, (n, k), device)?;
    let qt = QTensor::quantize(&rhs, dtype)?;
    let mm = QMatMul::from_qtensor(qt)?;
    let lhs = Tensor::rand(-1f32, 1f32, (m, k), device)?.to_dtype(act)?;

    for _ in 0..3 {
        let _ = mm.forward(&lhs)?;
    }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = mm.forward(&lhs)?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

/// The two decode launchers side by side on the same weights, bypassing the
/// dispatch in [`super::QRocmStorage::fwd`] (which only ever picks one of
/// them). This is what `MMVQ_MIN_WORK` is calibrated against.
#[test]
#[ignore = "benchmark: needs a GPU and takes seconds"]
fn bench_decode_paths() -> Result<()> {
    let device = rocm_device!();
    let dev = match &device {
        Device::Rocm(dev) => dev.clone(),
        _ => return Ok(()),
    };
    println!(
        "{:>6} {:>7} {:>6} {:>4} {:>10} {:>10}",
        "dtype", "k", "n", "m", "dmmv_ms", "mmvq_ms"
    );
    // SmolLM2-360M (hidden 960, ffn 2560, vocab 49152) and a 7B-ish shape.
    let shapes = [
        (960usize, 960usize),
        (960, 2560),
        (2560, 960),
        (960, 49152),
        (2048, 2048),
        (2048, 4096),
        (4096, 2048),
        (2048, 8192),
        (4096, 4096),
        (4096, 32000),
    ];
    for dtype in [
        GgmlDType::Q8_0,
        GgmlDType::Q4_0,
        GgmlDType::Q4K,
        GgmlDType::Q2K,
        GgmlDType::Q6K,
    ] {
        for (k, n) in shapes {
            if !k.is_multiple_of(dtype.block_size()) {
                continue;
            }
            let rhs = Tensor::rand(-1f32, 1f32, (n, k), &device)?;
            let qt = QTensor::quantize(&rhs, dtype)?;
            let q = match &qt.storage {
                crate::quantized::QStorage::Rocm(q) => q,
                _ => return Ok(()),
            };
            for m in [1usize, 4] {
                let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
                let y = dev.clone_htod(&lhs)?;
                let dmmv_ms = if m == 1 {
                    time(&device, 200, || {
                        super::dmmv::mul_mat_vec(&q.data, q.len, &y, 0, dtype, k, n, &dev)
                            .map(|_| ())
                    })?
                } else {
                    f64::NAN
                };
                let mmvq_ms = time(&device, 200, || {
                    super::mmvq::mul_mat_vec_via_q8_1(&q.data, q.len, &y, 0, dtype, k, n, m, &dev)
                        .map(|_| ())
                })?;
                println!(
                    "{:>6} {:>7} {:>6} {:>4} {:>10.4} {:>10.4}",
                    format!("{dtype:?}"),
                    k,
                    n,
                    m,
                    dmmv_ms,
                    mmvq_ms
                );
            }
        }
    }
    Ok(())
}

/// MMQ against the dense fallback it replaced, on the same weights.
///
/// This is what decides whether the MMQ dispatch in
/// [`super::QRocmStorage::fwd`] wants a size threshold under it: the dense path
/// materialises `n * k` weights in f32 whatever the batch, so the two only come
/// close when that matrix is small and the batch is large.
#[test]
#[ignore = "benchmark: needs a GPU and takes seconds"]
fn bench_prefill_paths() -> Result<()> {
    let device = rocm_device!();
    let dev = match &device {
        Device::Rocm(dev) => dev.clone(),
        _ => return Ok(()),
    };
    println!(
        "{:>6} {:>7} {:>6} {:>5} {:>10} {:>10} {:>8} {:>8}",
        "dtype", "k", "n", "m", "mmq_ms", "dense_ms", "speedup", "spread"
    );
    // Weight elements per shape: 0.25, 0.5, 1, 1, 2, 4, 8, 16 and 125 Mi. The
    // crossover is taken to be a function of that product alone, so the sweep
    // steps it in halvings around where it lands and carries two different
    // (k, n) at 1 Mi to check that the product really is what decides.
    let shapes = [
        (512usize, 512usize),
        (512, 1024),
        (1024, 1024),
        (512, 2048),
        (1024, 2048),
        (2048, 2048),
        (2048, 4096),
        (4096, 4096),
        (4096, 32000),
    ];
    for dtype in super::tests_mmq::MMQ_DTYPES {
        for (k, n) in shapes {
            let rhs = Tensor::rand(-1f32, 1f32, (n, k), &device)?;
            let qt = QTensor::quantize(&rhs, dtype)?;
            let deq = qt.dequantize(&device)?.t()?;
            let q = match &qt.storage {
                crate::quantized::QStorage::Rocm(q) => q,
                _ => return Ok(()),
            };
            for m in [9usize, 32, 128, 512] {
                let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
                let y = dev.clone_htod(&lhs)?;
                let lhs = Tensor::from_slice(&lhs, (m, k), &device)?;
                let iters = if m > 64 { 20 } else { 50 };
                let (mut mmq, mut dense, mut ratio) = (vec![], vec![], vec![]);
                for _ in 0..REPS {
                    let mmq_ms = time(&device, iters, || {
                        super::mmq::mul_mat_via_q8_1(&q.data, q.len, &y, 0, dtype, k, n, m, &dev)
                            .map(|_| ())
                    })?;
                    // The fallback's own computation: dequantize the weights,
                    // then GEMM against the transposed view.
                    let dense_ms = time(&device, iters, || {
                        let deq = qt.dequantize(&device)?.t()?;
                        lhs.matmul(&deq).map(|_| ())
                    })?;
                    mmq.push(mmq_ms);
                    dense.push(dense_ms);
                    ratio.push(dense_ms / mmq_ms);
                }
                let speedup = median(&mut ratio);
                // `median` sorted `ratio`, so its ends are the extremes.
                let spread = 100. * (ratio[REPS - 1] - ratio[0]) / speedup;
                println!(
                    "{:>6} {:>7} {:>6} {:>5} {:>10.4} {:>10.4} {:>8.2} {:>7.1}%",
                    format!("{dtype:?}"),
                    k,
                    n,
                    m,
                    median(&mut mmq),
                    median(&mut dense),
                    speedup,
                    spread
                );
            }
            drop(deq);
        }
    }
    Ok(())
}

/// Timing pairs per configuration in [`bench_prefill_paths`]. The two paths are
/// timed back to back inside a repeat, so a clock or occupancy drift moves both
/// and largely cancels in the ratio; the spread of the ratio across repeats is
/// what says whether a speedup near 1.0 is a real one.
pub(super) const REPS: usize = 5;

/// Median of `xs`, which it sorts in place.
pub(super) fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

/// Mean wall time of `run` over `iters` iterations, in milliseconds.
pub(super) fn time(
    device: &Device,
    iters: usize,
    mut run: impl FnMut() -> Result<()>,
) -> Result<f64> {
    for _ in 0..3 {
        run()?;
    }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..iters {
        run()?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

/// Every path end to end, through `QMatMul::forward` and so through the real
/// dispatch — unlike [`bench_prefill_roofline`], which launches the candidates
/// side by side to compare them. This is the one that says what a model
/// actually pays, and the one to re-run after touching the dispatch order.
#[test]
#[ignore = "benchmark: needs a GPU and takes seconds"]
fn bench_quantized_matmul() -> Result<()> {
    let device = rocm_device!();
    // A 4096x4096 projection and a 32000x4096 lm_head, the two shapes that
    // dominate a decode step, then Qwen3.5-2B's MLP either way round — which
    // is where the prefill batches below land.
    let shapes = [
        (4096usize, 4096usize),
        (4096, 32000),
        (2048, 8192),
        (8192, 2048),
    ];
    println!(
        "{:>6} {:>6} {:>7} {:>6} {:>5} {:>10}",
        "dtype", "m", "k", "n", "act", "ms"
    );
    for dtype in [GgmlDType::Q4_0, GgmlDType::Q4K, GgmlDType::Q6K] {
        for (k, n) in shapes {
            for (m, act) in [
                (1usize, DType::F32),
                (1, DType::F16),
                (4, DType::F32),
                (8, DType::F32),
                (64, DType::F32),
                // Prefill, in the dtype a served model runs: 1024 is one
                // llama-benchy `--pp` chunk, and 256 straddles
                // `dense::MIN_BATCH`.
                (255, DType::F16),
                (256, DType::F16),
                (1024, DType::F16),
                (1024, DType::F32),
            ] {
                let iters = if m > 8 { 10 } else { 50 };
                let ms = time_matmul(&device, dtype, m, k, n, act, iters)?;
                println!(
                    "{:>6} {:>6} {:>7} {:>6} {:>5} {:>10.4}",
                    format!("{dtype:?}"),
                    m,
                    k,
                    n,
                    format!("{act:?}"),
                    ms
                );
            }
        }
    }
    Ok(())
}
