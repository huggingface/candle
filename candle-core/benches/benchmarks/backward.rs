use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{Device, Var};
use criterion::{criterion_group, Criterion};
use std::hint::black_box;
use std::time::Instant;

fn run_backward_benchmark(c: &mut Criterion, device: &Device, name: &str) {
    let x = Var::randn(0.0, 1.0, &[128, 128, 64], device).unwrap();
    let y = Var::randn(0.0, 1.0, &[128, 64, 128], device).unwrap();
    let x = x.as_tensor();
    let y = y.as_tensor();
    let z = (x.matmul(&y)).unwrap();
    let z = ((&z + 0.5 * &z).unwrap() + 0.5).unwrap();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                let grads = black_box(&z).backward().unwrap();
                black_box(grads);
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
        let name = format!("backward");
        run_backward_benchmark(c, &device, &name);
    }
}

criterion_group!(benches, criterion_benchmark);
