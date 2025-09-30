use crate::benchmarks::{bench_device, BenchDevice};
use candle::{BackendStorage, DType, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};
use criterion::{black_box, criterion_group, Criterion};
use std::time::Instant;

const B: usize = 1;
const C: usize = 1;
const M: usize = 128;
const K: usize = 128;
const K_SIZE: usize = 3;

fn run<B: BackendStorage>(
    input: Tensor<B>,
    weight: Tensor<B>,
    bias: Tensor<B>,
    config: Conv2dConfig,
) {
    Conv2d::new(weight, Some(bias), config)
        .forward(&input)
        .unwrap();
}

fn run_conv2d_benchmark<B, D>(c: &mut Criterion, device: &D, dtype: DType, name: &str)
where
    B: BackendStorage<Device = D>,
    D: BenchDevice<B>,
{
    let weight: Tensor<B> = Tensor::ones((1, 1, K_SIZE, K_SIZE), dtype, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bias = Tensor::zeros(K, dtype, device).unwrap();
    let input = Tensor::ones((B, C, M, K), dtype, device).unwrap();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(
                    black_box(input.clone()),
                    black_box(weight.clone()),
                    black_box(bias.clone()),
                    Default::default(),
                );
            }
            device.synchronize().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let d = bench_device();
    run_conv2d_benchmark(c, &d, DType::F32, "conv2d_f32");
    run_conv2d_benchmark(c, &d, DType::F16, "conv2d_f16");
}

criterion_group!(benches, criterion_benchmark);
