use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run(a: &Tensor, b: &Tensor) {
    a.matmul(&b.t().unwrap()).unwrap();
}

fn run_bench(c: &mut Criterion, device: &Device, dtype: DType) {
    let b = 1;
    let m = 1;
    let n = 2048;
    let k = 2048;

    let lhs = Tensor::zeros((b, m, k), dtype, device).unwrap();
    let rhs = Tensor::zeros((b, n, k), dtype, device).unwrap();

    let flops = b * m * n * k;

    let name = match dtype {
        DType::F32 => "matmul_f32",
        DType::F16 => "matmul_f16",
        DType::BF16 => "matmul_bf16",
        DType::U8 => "matmul_fp8",
        _ => unimplemented!("{dtype:?} matmul bench not implemented"),
    };
    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&lhs), black_box(&rhs));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    let dtypes = vec![DType::F32, DType::F16, DType::BF16, DType::U8];
    for device in handler.devices {
        for dtype in dtypes.clone() {
            run_bench(c, &device, dtype);
        }
    }
}

criterion_group!(benches, criterion_benchmark);
