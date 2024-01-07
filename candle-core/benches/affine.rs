mod bench_utils;

use bench_utils::{bench_name, device, BenchDevice};
use candle_core::{DType, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::time::Instant;

fn run(a: &Tensor) {
    a.affine(12.34, 56.78).unwrap();
}

fn run_affine_benchmark(c: &mut Criterion, dtype: DType, name: &str) {
    let b = 1;
    let m = 1;
    let n = 2048;
    let k = 2048;

    let device = device().unwrap();
    let tensor = Tensor::zeros((b, m, k), dtype, &device).unwrap();

    let flops = b * m * n * k;

    let mut group = c.benchmark_group(bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&tensor));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    run_affine_benchmark(c, DType::F32, "affine_f32");
    run_affine_benchmark(c, DType::F16, "affine_f16");
    run_affine_benchmark(c, DType::BF16, "affine_bf16");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
