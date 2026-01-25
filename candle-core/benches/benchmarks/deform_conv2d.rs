use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

fn run(input: &Tensor, offset: &Tensor, weight: &Tensor, mask: Option<&Tensor>) {
    input
        .deform_conv2d(
            offset,
            weight,
            mask,
            None,   // bias
            (1, 1), // stride
            (1, 1), // padding
            (1, 1), // dilation
            1,      // groups
            1,      // offset_groups
        )
        .unwrap();
}

fn run_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    // Use different sizes for CPU vs GPU/Metal
    // CPU is much slower, so use smaller tensors
    let (channels, size) = match device {
        Device::Cpu => (64, 32), // Smaller for CPU: [1, 64, 32, 32]
        _ => (256, 64),          // Larger for GPU/Metal: [1, 256, 64, 64]
    };

    let input = Tensor::randn(0.0f32, 1.0, (1, channels, size, size), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let weight = Tensor::randn(0.0f32, 0.1, (channels, channels, 3, 3), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let offset = Tensor::randn(0.0f32, 0.5, (1, 18, size, size), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let mask = Tensor::rand(0.0f32, 1.0, (1, 9, size, size), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    // Calculate approximate FLOPs for deform_conv2d
    // deform_im2col + matmul
    let flops = input.dims().iter().product::<usize>() * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(
                    black_box(&input),
                    black_box(&offset),
                    black_box(&weight),
                    Some(black_box(&mask)),
                );
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
        run_benchmark(c, &device, DType::F32, "deform_conv2d_f32");
        // F16/BF16 benchmarks - skip on CPU as it's very slow
        if !matches!(device, Device::Cpu) {
            run_benchmark(c, &device, DType::F16, "deform_conv2d_f16");
            run_benchmark(c, &device, DType::BF16, "deform_conv2d_bf16");
        }
    }
}

criterion_group!(benches, criterion_benchmark);
