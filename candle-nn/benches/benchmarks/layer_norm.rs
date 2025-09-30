use crate::benchmarks::{bench_device, BenchDevice};
use candle::{BackendStorage, DType, Module, Tensor};
use candle_nn::LayerNorm;
use criterion::{black_box, criterion_group, Criterion};
use std::time::Instant;

fn run<B: BackendStorage>(input: &Tensor<B>, weight: &Tensor<B>, bias: &Tensor<B>) {
    let _ = LayerNorm::new(weight.clone(), bias.clone(), 1e-5).forward(input);
}

const B: usize = 1;
const M: usize = 1024;
const K: usize = 1024;

fn run_layer_norm_benchmark<B, D>(c: &mut Criterion, device: &D, dtype: DType, name: &str)
where
    B: BackendStorage<Device = D>,
    D: BenchDevice<B>,
{
    let elements = B * M * K;

    let weight: Tensor<B> = Tensor::arange(0.0, elements as f32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bias = weight.ones_like().unwrap();
    let input = weight.ones_like().unwrap();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&input), black_box(&weight), black_box(&bias));
            }
            device.synchronize().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let d = bench_device();
    run_layer_norm_benchmark(c, &d, DType::F32, "layer_norm_f32");
    run_layer_norm_benchmark(c, &d, DType::BF16, "layer_norm_bf16");
    run_layer_norm_benchmark(c, &d, DType::F16, "layer_norm_f16");
}

criterion_group!(benches, criterion_benchmark);
