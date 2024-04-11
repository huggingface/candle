use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{
    quantized::{self, GgmlDType, QMatMul},
    Device, Module, Tensor,
};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run(matmul: &QMatMul, x: &Tensor) {
    matmul.forward(&x).unwrap();
}

fn run_bench(c: &mut Criterion, device: &Device, dtype: GgmlDType) {
    let b = 1;
    let m = 1;
    let n = 2048;
    let k = 2048;

    let lhs = (0..(m * k))
        .map(|v| v as f32 / (m * k) as f32)
        .collect::<Vec<_>>();
    let rhs = (0..(k * n))
        .map(|v| v as f32 / (n * k) as f32)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_slice(&lhs, (m, k), device).unwrap();
    let rhs = Tensor::from_slice(&rhs, (k, n), device).unwrap();

    let qtensor = quantized::QTensor::quantize(&rhs.t().unwrap(), dtype).unwrap();
    let matmul = quantized::QMatMul::from_qtensor(qtensor).unwrap();

    let flops = b * m * n * k;

    let mut group = c.benchmark_group(device.bench_name("qmatmul"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&matmul), black_box(&lhs));
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
        for dtype in vec![GgmlDType::Q4K] {
            run_bench(c, &device, dtype);
        }
    }
}

criterion_group!(benches, criterion_benchmark);
