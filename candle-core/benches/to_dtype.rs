//! CPU conversion benchmarks. Input setup is outside timing; result allocation
//! and destruction are included. Use identical flags and Cargo.lock across revisions.
use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

fn conversions(c: &mut Criterion) {
    for (src, dst) in [
        (DType::F32, DType::F16),
        (DType::F16, DType::F32),
        (DType::F32, DType::BF16),
        (DType::BF16, DType::F32),
    ] {
        let mut group = c.benchmark_group(format!("cpu_to_dtype/{src:?}_to_{dst:?}"));
        for n in [64, 1024, 65_536, 1_048_576, 8_388_608] {
            let data: Vec<f32> = (0..n + 1)
                .map(|i| ((i * 13 % 131_071) as f32 - 65_535.) / 128.)
                .collect();
            let base = Tensor::from_vec(data, n + 1, &Device::Cpu)
                .unwrap()
                .to_dtype(src)
                .unwrap();
            let contiguous = base.narrow(0, 1, n).unwrap();
            let matrix = contiguous.reshape((n / 64, 64)).unwrap();
            let transposed = matrix.t().unwrap();
            let blocks = matrix.narrow(1, 1, 32).unwrap();
            for (name, tensor) in [
                ("contiguous", &contiguous),
                ("transposed", &transposed),
                ("blocks", &blocks),
            ] {
                group.throughput(Throughput::Elements(tensor.elem_count() as u64));
                group.bench_with_input(BenchmarkId::new(name, n), tensor, |b, tensor| {
                    b.iter(|| black_box(black_box(tensor).to_dtype(black_box(dst)).unwrap()))
                });
            }
        }
        group.finish();
    }
}

criterion_group!(benches, conversions);
criterion_main!(benches);
