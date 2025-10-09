use crate::benchmarks::{bench_device, BenchDevice};
use candle::{BackendStorage, DType, Tensor};
use candle_nn::ops::softmax_last_dim;
use criterion::Throughput;
use criterion::{black_box, criterion_group, Criterion};
use std::time::Instant;

fn run<B: BackendStorage>(input: &Tensor<B>) {
    let _ = softmax_last_dim(&input).unwrap();
}

const B: usize = 1;
const M: usize = 1024;
const K: usize = 1024;

fn run_softmax_benchmark<B, D>(c: &mut Criterion, device: &D, dtype: DType, name: &str)
where
    B: BackendStorage<Device = D>,
    D: BenchDevice<B>,
{
    let elements = B * M * K;

    let input: Tensor<B> = Tensor::rand(-1000.0f32, 1000.0f32, (B, M, K), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let flops = elements * dtype.size_in_bytes();
    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&input));
            }
            device.synchronize().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let d = bench_device();
    run_softmax_benchmark(c, &d, DType::F32, "softmax_f32");
    run_softmax_benchmark(c, &d, DType::BF16, "softmax_bf16");
    run_softmax_benchmark(c, &d, DType::F16, "softmax_f16");
}

criterion_group!(benches, criterion_benchmark);
