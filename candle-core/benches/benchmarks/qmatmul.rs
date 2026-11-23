#[path = "qmatmul/report.rs"]
mod report;

use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{
    quantized::{self, GgmlDType, QMatMul},
    Device, Module, Tensor,
};
use criterion::{Criterion, Throughput};
use std::cell::Cell;
use std::hint::black_box;
use std::time::Instant;

fn run(matmul: &QMatMul, x: &Tensor) {
    matmul.forward(x).unwrap();
}

fn run_bench(c: &mut Criterion, device: &Device, dtype: GgmlDType) {
    let b = 1;
    let m = 1;
    let n = 1024;
    let k = 1024;

    let lhs = (0..(m * k))
        .map(|v| v as f32 / (m * k) as f32)
        .collect::<Vec<_>>();
    let rhs = (0..(k * n))
        .map(|v| v as f32 / (n * k) as f32)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_slice(&lhs, (m, k), device).unwrap();
    let rhs = Tensor::from_slice(&rhs, (k, n), device).unwrap();

    let qtensor = quantized::QTensor::quantize(&rhs.t().unwrap(), dtype).unwrap();
    let weight_bytes = qtensor.storage_size_in_bytes() as u64;
    let matmul = quantized::QMatMul::from_qtensor(qtensor).unwrap();

    let flops = b * m * n * k;

    let benchmark_name = device.bench_name(format!("qmatmul_{dtype:?}"));
    let family_name = device.bench_name("qmatmul");
    let mut group = c.benchmark_group(&benchmark_name);
    group.sample_size(200);
    group.throughput(Throughput::Bytes(flops as u64));
    let measured = Cell::new(false);
    group.bench_function("iter", |b| {
        measured.set(true);
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

    if measured.get() {
        // A matmul does one multiply and one add per inner-dimension element.
        report::record(
            &benchmark_name,
            &family_name,
            &format!("{dtype:?}"),
            &format!("lhs {m}x{k}, rhs {k}x{n}"),
            2 * flops as u64,
            weight_bytes,
            (k * n) as u64,
        );
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            run_bench(c, &device, dtype);
        }
    }
}

// The expansion of `criterion_group!`, written out so the report can draw once
// the whole dtype sweep has reported — the boundary the macro leaves no room for.
pub fn benches() {
    let mut criterion = Criterion::default().configure_from_args();
    criterion_benchmark(&mut criterion);
    report::print_summary();
}
