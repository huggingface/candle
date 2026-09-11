//! The prefill half of the quantized timing harness: MMQ against the roofline,
//! and against the dense path that now pre-empts it.
//!
//! Split from [`super::bench`], which is at the workspace 400-line cap. Same
//! invocation — both files are `quantized::rocm::bench*`:
//!
//! ```text
//! cargo test -p candle-core --features rocm --lib --release -- \
//!     quantized::rocm::bench --ignored --nocapture --test-threads=1
//! ```

use super::bench::{median, time, REPS};
use crate::quantized::{GgmlDType, QTensor};
use crate::rocm_backend::RocmDevice;
use crate::{DType, Device, Result, Tensor};

macro_rules! rocm_device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => Device::Rocm(dev),
            Err(_) => return Ok(()),
        }
    };
}

/// Peak VRAM bandwidth of the card the tables in [`super::mmq`] were measured
/// on (RX 7800 XT, gfx1101: 19.5 Gbps GDDR6 on a 256-bit bus). Only a yardstick
/// for the printout below — nothing dispatches on it.
const PEAK_GBPS: f64 = 624.1;

/// MMQ at prefill batch sizes against the roofline, and against the dense path
/// that now pre-empts it. Shapes are Qwen3.5-2B's (hidden 2048, ffn 8192).
///
/// `super::bench::bench_prefill_paths` stops at `m = 512` and only answers "is
/// MMQ faster than dequantize + GEMM", a comparison against something slow.
/// This asks the absolute question at a real prompt's batch:
///
/// * `w_GB/s` — weight bytes per second, counting the `ceil(m / mmq_x)` passes
///   the kernel makes over the matrix. Against [`PEAK_GBPS`] it says how close
///   to DRAM-bound it runs; far above peak is the 64 MB Infinity Cache
///   absorbing the re-reads, not a bad measurement.
/// * `nnTF/s` — `2*m*k*n` over the dense path's wall time.
///
/// `tn_ms` is the dequantized `(n, k)` buffer described with swapped strides,
/// as the fallback used to; `nn_ms` is [`super::dense::forward`], which
/// reorients first. `v_mmq` above 1.0 means the dense path wins, and is what
/// `dense::MIN_BATCH` is set from; `v_tn` above 1.0 means the reorientation
/// paid for itself, which it often does not — see [`super::dense`].
#[test]
#[ignore = "benchmark: needs a GPU and takes seconds"]
fn bench_prefill_roofline() -> Result<()> {
    let device = rocm_device!();
    let dev = match &device {
        Device::Rocm(dev) => dev.clone(),
        _ => return Ok(()),
    };
    let tiles = dev.mmq_tiles();
    println!("mmq tiles: {tiles:?}, peak {PEAK_GBPS} GB/s");
    println!(
        "{:>6} {:>7} {:>6} {:>6} {:>9} {:>8} {:>8} {:>7} {:>7} {:>8} {:>8}",
        "dtype", "k", "n", "m", "mmq_ms", "tn_ms", "nn_ms", "v_mmq", "v_tn", "w_GB/s", "nnTF/s"
    );
    // (k, n): qkv and o_proj at 2048x2048, then the MLP's 2048x8192 up/gate and
    // its 8192x2048 down.
    let shapes = [
        (2048usize, 2048usize),
        (2048, 4096),
        (2048, 8192),
        (8192, 2048),
    ];
    for dtype in [GgmlDType::Q8_0, GgmlDType::Q4K, GgmlDType::Q4_0] {
        let mmq_x = match super::mmq::geometry(dtype, tiles) {
            Some((mmq_x, _, _)) => mmq_x,
            None => continue,
        };
        for (k, n) in shapes {
            let rhs = Tensor::rand(-1f32, 1f32, (n, k), &device)?;
            let qt = QTensor::quantize(&rhs, dtype)?;
            let q = match &qt.storage {
                crate::quantized::QStorage::Rocm(q) => q,
                _ => return Ok(()),
            };
            let w_bytes = n * k / dtype.block_size() * dtype.type_size();
            for m in [128usize, 256, 384, 512, 1024, 2048] {
                let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
                let y = dev.clone_htod(&lhs)?;
                let lhs = Tensor::from_slice(&lhs, (m, k), &device)?;
                let lhs16 = lhs.to_dtype(DType::F16)?;
                let (act, act_l) = lhs16.storage_and_layout();
                let act = match &*act {
                    crate::Storage::Rocm(s) => s,
                    _ => return Ok(()),
                };
                let (mut mmq, mut tn, mut nn) = (vec![], vec![], vec![]);
                for _ in 0..REPS {
                    mmq.push(time(&device, 20, || {
                        super::mmq::mul_mat_via_q8_1(&q.data, q.len, &y, 0, dtype, k, n, m, &dev)
                            .map(|_| ())
                    })?);
                    // The orientation this path used to hand rocBLAS: the
                    // dequantized `(n, k)` buffer described with swapped
                    // strides, which reaches `gemm_config` as a transpose.
                    tn.push(time(&device, 20, || {
                        let deq = qt.dequantize_f16(&device)?.t()?;
                        lhs16.matmul(&deq).map(|_| ())
                    })?);
                    // The whole fallback as `fwd` runs it now: f16 dequantize,
                    // reorient, plain GEMM.
                    nn.push(time(&device, 20, || {
                        super::dense::forward(q, (1, m, n, k), act, act_l).map(|_| ())
                    })?);
                }
                let (mmq_ms, tn_ms, nn_ms) = (median(&mut mmq), median(&mut tn), median(&mut nn));
                let secs = mmq_ms * 1e-3;
                // The kernel re-reads the whole weight matrix once per column
                // tile, so this is traffic generated, not bytes touched.
                let traffic = (m.div_ceil(mmq_x) * w_bytes) as f64;
                let flops = 2. * (m * k * n) as f64;
                println!(
                    "{:>6} {:>7} {:>6} {:>6} {:>9.4} {:>8.4} {:>8.4} {:>7.2} {:>7.2} {:>8.1} {:>8.2}",
                    format!("{dtype:?}"),
                    k,
                    n,
                    m,
                    mmq_ms,
                    tn_ms,
                    nn_ms,
                    mmq_ms / nn_ms,
                    tn_ms / nn_ms,
                    traffic / secs / 1e9,
                    flops / (nn_ms * 1e-3) / 1e12
                );
            }
        }
    }
    Ok(())
}
