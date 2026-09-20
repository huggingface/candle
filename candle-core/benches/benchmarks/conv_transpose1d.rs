use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, BenchmarkId, Criterion};
use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    let device = &Device::Cpu;
    let mut group = c.benchmark_group("cpu_conv_transpose1d");
    for (c_in, c_out, len, kernel, stride) in [
        (64, 32, 8, 4, 2),
        (128, 64, 3000, 8, 4),
        (16, 64, 2048, 7, 1),
        (16, 64, 2048, 4, 4),
        (16, 64, 2048, 3, 5),
        (16, 64, 2048, 1, 1),
    ] {
        let input = Tensor::ones((1, c_in, len), DType::F32, device).unwrap();
        let weights = Tensor::ones((c_in, c_out, kernel), DType::F32, device).unwrap();
        let id = BenchmarkId::new(
            format!("{c_in}x{c_out}x{len}"),
            format!("k{kernel}_s{stride}"),
        );
        group.bench_function(id, |b| {
            b.iter(|| {
                black_box(&input)
                    .conv_transpose1d(black_box(&weights), 0, 0, stride, 1, 1)
                    .unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
