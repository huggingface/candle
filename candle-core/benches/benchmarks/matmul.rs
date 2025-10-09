use crate::benchmarks::{bench_device, BenchDevice};
use candle_core::{BackendStorage, DType, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run<B: BackendStorage>(a: &Tensor<B>, b: &Tensor<B>) {
    a.matmul(&b.t().unwrap()).unwrap();
}

fn run_bench<B, D>(c: &mut Criterion, device: &D)
where
    B: BackendStorage<Device = D>,
    D: BenchDevice<B>,
{
    let b = 1;
    let m = 1;
    let n = 2048;
    let k = 2048;

    let dtype = DType::F32;
    let lhs: Tensor<B> = Tensor::zeros((b, m, k), dtype, device).unwrap();
    let rhs = Tensor::zeros((b, n, k), dtype, device).unwrap();

    let flops = b * m * n * k;

    let mut group = c.benchmark_group(device.bench_name("matmul"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&lhs), black_box(&rhs));
            }
            device.synchronize().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = bench_device();
    run_bench(c, &device);
}

criterion_group!(benches, criterion_benchmark);
