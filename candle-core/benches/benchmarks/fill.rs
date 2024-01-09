use crate::benchmarks::{bench_name, device, BenchDevice};
use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run(shape: (usize, usize, usize), dtype: DType, device: &Device) {
    Tensor::ones(shape, dtype, device).unwrap();
}

fn run_fill_benchmark(c: &mut Criterion, name: &str, dtype: DType) {
    let b = 1;
    let rows = 4096;
    let columns = 4096;

    let flops = b * rows * columns * dtype.size_in_bytes();

    let device = device().unwrap();

    let mut group = c.benchmark_group(bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |bencher| {
        bencher.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(
                    black_box((b, rows, columns)),
                    black_box(dtype),
                    black_box(&device),
                );
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    run_fill_benchmark(c, "fill_u8", DType::U8);
    run_fill_benchmark(c, "fill_f32", DType::F32);
}

criterion_group!(benches, criterion_benchmark);
