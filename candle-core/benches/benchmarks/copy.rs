use crate::benchmarks::{bench_device, BenchDevice};
use candle_core::{BackendStorage, Tensor, WithDType};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run_copy_mask_benchmark<DType: WithDType, B, D>(c: &mut Criterion, device: &D, name: &str)
where
    B: BackendStorage<Device = D>,
    D: BenchDevice<B>,
{
    let batch_size = 128;
    let in_seq_len = 1;
    let kv_seq_len = 1024;

    let attn_mask = vec![vec![vec![DType::zero(); kv_seq_len]; in_seq_len]; batch_size];
    let size_in_bytes = batch_size * in_seq_len * kv_seq_len * DType::DTYPE.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(size_in_bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let attn_masks = vec![attn_mask.clone(); iters as usize];
            let start = Instant::now();
            for attn_mask in attn_masks.into_iter() {
                let tensor: Tensor<B> = Tensor::new(black_box(attn_mask), device).unwrap();
                black_box(tensor);
            }
            device.synchronize().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = bench_device();
    run_copy_mask_benchmark::<f32, _, _>(c, &device, "copy_mask");
}

criterion_group!(benches, criterion_benchmark);
