use candle_core::{DType, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;
use crate::benchmarks::{bench_name, device, BenchDevice};

fn run(a: &Tensor) {
    a.sum(2).unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    let b = 1;
    let m = 2048;
    let k = 2048;

    let device = device().unwrap();

    let a = Tensor::rand(-1000.0f32, 1000.0f32, (b, m, k), &device).unwrap();

    let flops = b * m * k * DType::F32.size_in_bytes();

    let mut group = c.benchmark_group(bench_name("reduce"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&a));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
