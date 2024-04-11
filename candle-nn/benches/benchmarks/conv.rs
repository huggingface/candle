use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};
use criterion::{black_box, criterion_group, Criterion};
use std::time::Instant;

const B: usize = 1;
const C: usize = 1;
const M: usize = 128;
const K: usize = 128;
const K_SIZE: usize = 3;

fn run(input: Tensor, weight: Tensor, bias: Tensor, config: Conv2dConfig) {
    Conv2d::new(weight, Some(bias), config)
        .forward(&input)
        .unwrap();
}

fn run_conv2d_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let weight = Tensor::ones((1, 1, K_SIZE, K_SIZE), dtype, device)
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
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = BenchDeviceHandler::new().unwrap();
    for d in device.devices {
        run_conv2d_benchmark(c, &d, DType::F32, "conv2d_f32");
        run_conv2d_benchmark(c, &d, DType::F16, "conv2d_f16");
    }
}

criterion_group!(benches, criterion_benchmark);
