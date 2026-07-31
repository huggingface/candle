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

/// Mean wall time of `run` over `iters` iterations, in milliseconds.
fn time(device: &Device, iters: usize, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
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

#[test]
#[ignore = "benchmark: needs a GPU and takes seconds"]
fn bench_quantized_matmul() -> Result<()> {
    let device = rocm_device!();
    // A 4096x4096 projection and a 32000x4096 lm_head, the two shapes that
    // dominate a decode step.
    let shapes = [(4096usize, 4096usize), (4096, 32000)];
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
