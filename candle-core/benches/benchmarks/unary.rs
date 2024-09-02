use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run(a: &Tensor) {
    a.sqrt().unwrap();
}

fn run_unary_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let b = 1;
    let m = 1024;
    let k = 1024;

    let tensor = Tensor::arange(0.0f32, (b * m * k) as f32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .reshape((b, m, k))
        .unwrap();

    let flops = b * m * k * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
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
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let name = format!("sqrt_{:?}", dtype);
            run_unary_benchmark(c, &device, dtype, &name);
        }
    }
}

criterion_group!(benches, criterion_benchmark);
