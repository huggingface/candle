use crate::benchmarks::{bench_name, device, BenchDevice};
use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn rand_uniform(a: &Tensor) {
    a.rand_like(-1.0, 123.0).unwrap();
}

fn rand_normal(a: &Tensor) {
    a.randn_like(100.0, 15.0).unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    let b = 1;

    let rows = 2048;
    let cols = 2048;

    let d = device().unwrap();
    let dtype = DType::F32;
    let tensor = Tensor::zeros((b, rows, cols), dtype, &d).unwrap();

    let flops = b * rows * cols * dtype.size_in_bytes();

    let mut group = c.benchmark_group(bench_name("random_uniform"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |benches| {
        benches.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                rand_uniform(black_box(&tensor));
            }
            d.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();

    let d = device().unwrap();
    let tensor = Tensor::zeros((b, rows, cols), dtype, &d).unwrap();

    let mut group = c.benchmark_group(bench_name("random_normal"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |benches| {
        benches.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                rand_normal(black_box(&tensor));
            }
            d.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
