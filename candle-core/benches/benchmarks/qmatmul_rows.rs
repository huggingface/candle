//! A quantized matmul with a few rows of input `[m, 1, k]`, as in the decode
//! step of a batch of `m` sequences. The shapes are those of Qwen2.5-0.5B.
//!
//! The weights come from a ring of copies of 256 MiB or more, so that each
//! op reads its weights from memory, as in a decode step. With one copy, a
//! small weight matrix stays in the GPU cache and the op is faster than in a
//! model.
//!
//! Run one format and row count with a filter, for example:
//! `cargo bench -p candle-core --features metal --bench bench_main -- qmatmul_rows_Q4_0/down_m8`
use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{
    quantized::{GgmlDType, QMatMul, QStorage, QTensor},
    Device, Module, Tensor,
};
use criterion::{criterion_group, Criterion};
use std::hint::black_box;
use std::time::Instant;

const RING_BYTES: usize = 256 << 20;

// (name, n, k): the matmuls of one Qwen2.5-0.5B decode step. o has the
// shape of q.
const SHAPES: [(&str, usize, usize); 5] = [
    ("q", 896, 896),
    ("k_v", 128, 896),
    ("gate_up", 4864, 896),
    ("down", 896, 4864),
    ("lm_head", 151936, 896),
];

fn run_bench(c: &mut Criterion, device: &Device, dtype: GgmlDType) {
    let mut group = c.benchmark_group(device.bench_name(format!("qmatmul_rows_{dtype:?}")));
    group.sample_size(50);
    for (name, n, k) in SHAPES {
        let w = (Tensor::randn(0f32, 1.0, (n, k), &Device::Cpu).unwrap() * 0.02).unwrap();
        let q = QTensor::quantize(&w, dtype).unwrap();
        let data = q.data().unwrap();
        let copies = RING_BYTES.div_ceil(data.len()).max(2);
        let ring: Vec<QMatMul> = (0..copies)
            .map(|_| {
                let storage = QStorage::from_data(data.clone(), device, dtype).unwrap();
                QMatMul::from_qtensor(QTensor::new(storage, (n, k)).unwrap()).unwrap()
            })
            .collect();
        for m in [1, 2, 4, 8, 16] {
            let x = Tensor::randn(0f32, 1.0, (m, 1, k), device).unwrap();
            let ring = &ring;
            group.bench_function(format!("{name}_m{m}"), move |b| {
                b.iter_custom(|iters| {
                    device.sync().unwrap();
                    let start = Instant::now();
                    for i in 0..iters as usize {
                        black_box(ring[i % ring.len()].forward(black_box(&x)).unwrap());
                    }
                    device.sync().unwrap();
                    start.elapsed()
                })
            });
        }
    }
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        if !device.is_metal() {
            continue;
        }
        for dtype in [GgmlDType::Q4_0, GgmlDType::Q8_0] {
            run_bench(c, &device, dtype);
        }
    }
}

criterion_group!(benches, criterion_benchmark);
