use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::time::Instant;

fn rand_uniform(a: &Tensor) {
    a.rand_like(0.0, 1.0).unwrap();
}

fn rand_normal(a: &Tensor) {
    a.randn_like(100.0, 15.0).unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    let b = 1;

    let rows = 2048;
    let cols = 2048;

    let device = Device::new_metal(0).unwrap();
    let device2 = device.clone();
    let dtype = DType::F32;
    let tensor = Tensor::zeros((b, rows, cols), dtype, &device).unwrap();

    let flops = b * rows * cols;

    let mut group = c.benchmark_group("metal_random_uniform");
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |benches| {
        benches.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                rand_uniform(black_box(&tensor));
            }
            if let Device::Metal(device) = &device {
                device.wait_until_completed().unwrap();
            } else {
                panic!("Expected metal device");
            }
            start.elapsed()
        })
    });
    group.finish();

    let tensor = Tensor::zeros((b, rows, cols), dtype, &device2).unwrap();

    let mut group = c.benchmark_group("metal_random_normal");
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |benches| {
        benches.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                rand_normal(black_box(&tensor));
            }
            if let Device::Metal(device) = &device2 {
                device.wait_until_completed().unwrap();
            } else {
                panic!("Expected metal device");
            }
            start.elapsed()
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
