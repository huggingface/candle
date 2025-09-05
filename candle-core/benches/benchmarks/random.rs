use crate::benchmarks::{bench_device, BenchDevice};
use candle_core::{BackendStorage, DType, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn rand_uniform<B: BackendStorage>(a: &Tensor<B>) {
    a.rand_like(-1.0, 123.0).unwrap();
}

fn rand_normal<B: BackendStorage>(a: &Tensor<B>) {
    a.randn_like(100.0, 15.0).unwrap();
}

fn run_random_bench<B, D>(c: &mut Criterion, device: &D)
where
    B: BackendStorage<Device = D>,
    D: BenchDevice<B>,
{
    let b = 1;

    let rows = 2048;
    let cols = 2048;

    let dtype = DType::F64;
    let tensor: Tensor<B> = Tensor::zeros((b, rows, cols), dtype, device).unwrap();

    let flops = b * rows * cols * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name("random_uniform"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |benches| {
        benches.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                rand_uniform(black_box(&tensor));
            }
            device.synchronize().unwrap();
            start.elapsed()
        })
    });
    group.finish();

    let tensor: Tensor<B> = Tensor::zeros((b, rows, cols), dtype, device).unwrap();

    let mut group = c.benchmark_group(device.bench_name("random_normal"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |benches| {
        benches.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                rand_normal(black_box(&tensor));
            }
            device.synchronize().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = bench_device();
    run_random_bench(c, &device);
}

criterion_group!(benches, criterion_benchmark);
