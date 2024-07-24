use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run(input: &Tensor){
    candle_nn::ops::softmax(input, candle::D::Minus1).unwrap();
}

fn run_softmax_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str, (b, m, k) : (usize, usize, usize)) {
    let tensor = Tensor::arange(0.0f32, (b * m * k) as f32, &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .reshape((b, m, k))
        .unwrap();

    let flops = b * m * k * dtype.size_in_bytes();
    
    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("copy", move |b| {
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
            let name = format!("softmax_{:?}", dtype);
            if device.is_dtype_available(dtype){
                run_softmax_benchmark(c, &device, dtype, &name, (1, 1024, 1024));
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
