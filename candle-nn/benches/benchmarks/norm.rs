use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, RmsNorm};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

fn run_layer_norm(input: &Tensor, weight: &Tensor, bias: &Tensor) {
    let _ = LayerNorm::new(weight.clone(), bias.clone(), 1e-5).forward(input);
}

fn run_rms_norm(input: &Tensor, weight: &Tensor) {
    let _ = RmsNorm::new(weight.clone(), 1e-5).forward(input);
}

const B: usize = 1;
const M: usize = 1024;
const K: usize = 1024;

fn run_layer_norm_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let elements = B * M * K;

    let weight = Tensor::arange(0.0, elements as f32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bias = weight.ones_like().unwrap();
    let input = weight.ones_like().unwrap();

    let flops = elements * dtype.size_in_bytes();
    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_layer_norm(black_box(&input), black_box(&weight), black_box(&bias));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn run_rms_norm_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let elements = B * M * K;

    let weight = Tensor::arange(0.0, elements as f32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let input = weight.ones_like().unwrap();

    let flops = elements * dtype.size_in_bytes();
    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_rms_norm(black_box(&input), black_box(&weight));
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
        run_rms_norm_benchmark(c, &d, DType::F32, "rms_norm_f32");
        run_rms_norm_benchmark(c, &d, DType::BF16, "rms_norm_bf16");
        run_rms_norm_benchmark(c, &d, DType::F16, "rms_norm_f16");
        run_layer_norm_benchmark(c, &d, DType::F32, "layer_norm_f32");
        run_layer_norm_benchmark(c, &d, DType::BF16, "layer_norm_bf16");
        run_layer_norm_benchmark(c, &d, DType::F16, "layer_norm_f16");
    }
}

criterion_group!(benches, criterion_benchmark);
