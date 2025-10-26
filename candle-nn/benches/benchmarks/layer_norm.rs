use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Module};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

fn run_layer_norm(ln: &LayerNorm, input: &Tensor) {
    ln.forward(input).unwrap();
}

fn run_layer_norm_benchmark(
    c: &mut Criterion,
    device: &Device,
    dtype: DType,
    name: &str,
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
) {
    let input = Tensor::randn(0f32, 1f32, (batch, seq_len, hidden_size), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let weight = Tensor::ones(hidden_size, dtype, device).unwrap();
    let bias = Tensor::zeros(hidden_size, dtype, device).unwrap();
    let ln = LayerNorm::new(weight, bias, 1e-5);

    let flops = batch * seq_len * hidden_size * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_layer_norm(black_box(&ln), black_box(&input));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();

    // BERT-base size: batch=8, seq_len=128, hidden=768
    for device in handler.devices.iter() {
        run_layer_norm_benchmark(
            c,
            device,
            DType::F32,
            "layer_norm_bert_base_f32",
            8,
            128,
            768,
        );
        run_layer_norm_benchmark(
            c,
            device,
            DType::F16,
            "layer_norm_bert_base_f16",
            8,
            128,
            768,
        );
    }

    // BERT-large size: batch=8, seq_len=128, hidden=1024
    for device in handler.devices.iter() {
        run_layer_norm_benchmark(
            c,
            device,
            DType::F32,
            "layer_norm_bert_large_f32",
            8,
            128,
            1024,
        );
        run_layer_norm_benchmark(
            c,
            device,
            DType::F16,
            "layer_norm_bert_large_f16",
            8,
            128,
            1024,
        );
    }

    // Small size for quick iteration
    for device in handler.devices.iter() {
        run_layer_norm_benchmark(c, device, DType::F32, "layer_norm_small_f32", 2, 32, 256);
        run_layer_norm_benchmark(c, device, DType::F16, "layer_norm_small_f16", 2, 32, 256);
    }

    // Very large hidden size
    for device in handler.devices.iter() {
        run_layer_norm_benchmark(c, device, DType::F32, "layer_norm_xlarge_f32", 4, 64, 4096);
        run_layer_norm_benchmark(c, device, DType::F16, "layer_norm_xlarge_f16", 4, 64, 4096);
    }
}

criterion_group!(benches, criterion_benchmark);
