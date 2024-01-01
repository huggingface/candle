use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::time::Instant;

fn run(shape: (usize, usize, usize), dtype: DType, device: &Device) {
    Tensor::ones(shape, dtype, device).unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    let b = 1;
    let rows = 4096;
    let columns = 4096;

    let flops = b * rows * columns;

    let device1 = Device::new_metal(0).unwrap();
    let device2 = device1.clone();

    let mut group = c.benchmark_group("fill_metal_u8");
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |bencher| {
        bencher.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box((b, rows, columns)), black_box(DType::U8), black_box(&device1));
            }
            if let Device::Metal(device) = &device1 {
                device.wait_until_completed().unwrap();
            } else {
                panic!("Expected metal device");
            }
            start.elapsed()
        })
    });
    group.finish();

    let mut group = c.benchmark_group("fill_metal_f32");
    group.throughput(Throughput::Bytes((flops * DType::F32.size_in_bytes()) as u64));
    group.bench_function("iter", move |bencher| {
        bencher.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box((b, rows, columns)), black_box(DType::F32), black_box(&device2));
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
