mod bench_utils;

use crate::bench_utils::bench_name;
use bench_utils::{device, BenchDevice};
use candle_core::{DType, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::time::Instant;

fn run(a: &Tensor, b: &Tensor) {
    a.matmul(&b.t().unwrap()).unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    let b = 1;
    let m = 1;
    let n = 2048;
    let k = 2048;

    let device = device().unwrap();
    let dtype = DType::F32;
    let lhs = Tensor::zeros((b, m, k), dtype, &device).unwrap();
    let rhs = Tensor::zeros((b, n, k), dtype, &device).unwrap();

    let flops = b * m * n * k;

    let mut group = c.benchmark_group(bench_name("matmul"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&lhs), black_box(&rhs));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
