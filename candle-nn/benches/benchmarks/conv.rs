use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};
use criterion::{black_box, criterion_group, Criterion};
use std::time::Instant;

const B: usize = 1;
const C: usize = 1;

fn run(input: Tensor, weight: Tensor, bias: Option<Tensor>, config: Conv2dConfig) {
    Conv2d::new(weight, bias, config).forward(&input).unwrap();
}

fn run_conv2d_benchmark(
    c: &mut Criterion,
    device: &Device,
    dtype: DType,
    k_size: usize,
    m: usize,
    bias: bool,
    name: &str,
) {
    let weight = Tensor::ones((1, C, k_size, k_size), dtype, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bias = if bias {
        Some(Tensor::zeros(m, dtype, device).unwrap())
    } else {
        None
    };
    let input = Tensor::ones((B, C, m, m), dtype, device).unwrap();

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
        run_conv2d_benchmark(c, &d, DType::F32, 3, 128, true, "conv2d_f32_i128_k3x3_b");
        run_conv2d_benchmark(c, &d, DType::F32, 1, 128, false, "conv2d_f32_i128_k1x1_nb");
        run_conv2d_benchmark(c, &d, DType::F32, 5, 128, false, "conv2d_f32_i128_k5x5_nb");
        run_conv2d_benchmark(c, &d, DType::F32, 3, 512, false, "conv2d_f32_i512_k3x3_nb");
        run_conv2d_benchmark(c, &d, DType::F16, 3, 128, true, "conv2d_f16_i128_k3x3_b");
        run_conv2d_benchmark(c, &d, DType::F16, 1, 128, false, "conv2d_f16_i128_k1x1_nb");
        run_conv2d_benchmark(c, &d, DType::F16, 5, 128, false, "conv2d_f16_i128_k5x5_nb");
        run_conv2d_benchmark(c, &d, DType::F16, 5, 512, false, "conv2d_f16_i512_k3x3_nb");
    }
}

criterion_group!(benches, criterion_benchmark);
