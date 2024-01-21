use crate::benchmarks::{bench_name, device, BenchDevice};
use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run_sum(a: &Tensor) {
    a.sum(2).unwrap();
}
fn run_arg_min(a: &Tensor) {
    a.argmin(2).unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = device().unwrap();
    run_reduce(c, &device);
    run_arg_reduce(c, &device);
}
fn run_reduce(c: &mut Criterion, device: &Device) {
    let b = 1;
    let m = 2048;
    let k = 2048;

    let a = Tensor::rand(-1000.0f32, 1000.0f32, (b, m, k), &device).unwrap();

    let flops = b * m * k * DType::F32.size_in_bytes();

    let mut group = c.benchmark_group(bench_name("reduce"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_sum(black_box(&a));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn run_arg_reduce(c: &mut Criterion, device: &Device) {
    let b = 1;
    let m = 2048;
    let k = 2048;

    let a = Tensor::rand(-1000.0f32, 1000.0f32, (b, m, k), &device).unwrap();

    let flops = b * m * k * DType::F32.size_in_bytes();

    let mut group = c.benchmark_group(bench_name("arg_reduce"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_arg_min(black_box(&a));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
