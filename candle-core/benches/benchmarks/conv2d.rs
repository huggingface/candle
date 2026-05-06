use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

fn run(x: &Tensor, k: &Tensor, padding: usize, stride: usize, dilation: usize, groups: usize) {
    x.conv2d(k, padding, stride, dilation, groups).unwrap();
}

fn run_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let t = Tensor::arange(0.0f32, (1 * 128 * 28 * 28) as f32, device)
        .unwrap()
        .reshape((1, 128, 28, 28))
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let kernel = Tensor::arange(0.0f32, (256 * 128 * 3 * 3) as f32, device)
        .unwrap()
        .reshape((256, 128, 3, 3))
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let flops = t.dims().iter().product::<usize>() * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&t), black_box(&kernel), 1, 1, 1, 1);
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        run_benchmark(c, &device, DType::F32, "conv2d_f32");
    }
}

criterion_group!(benches, criterion_benchmark);
