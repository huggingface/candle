use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Tensor};
use candle_nn::scaled_dot_product_attention;
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run_attention(q: &Tensor, k: &Tensor, v: &Tensor, m: &Tensor, s: f64) {
    let att = (q
        .contiguous()
        .unwrap()
        .matmul(&k.t().unwrap().contiguous().unwrap())
        .unwrap()
        / s)
        .unwrap();

    let att = att.broadcast_add(m).unwrap();

    let att = candle_nn::ops::softmax_last_dim(&att).unwrap();
    // Convert to contiguous as matmul doesn't support strided vs for now.
    att.matmul(&v.contiguous().unwrap()).unwrap();
}

fn run_bench_naive(c: &mut Criterion, device: &Device) {
    let b = 4;
    let seq = 1024;
    let heads = 32;
    let hd = 128;

    let dtype = DType::F32;
    let q = Tensor::zeros((b, heads, seq, hd), dtype, device).unwrap();
    let k = Tensor::zeros((b, heads, seq, hd), dtype, device).unwrap();
    let v = Tensor::zeros((b, heads, seq, hd), dtype, device).unwrap();
    let m = Tensor::zeros((b, heads, seq, seq), dtype, device).unwrap();

    let flops = b * seq * heads * hd;

    let mut group = c.benchmark_group(device.bench_name("attention_naive"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    black_box(&m),
                    0.3,
                );
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark_naive(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        run_bench_naive(c, &device);
    }
}

fn run_bench_fast(c: &mut Criterion, device: &Device) {
    let b = 4;
    let seq = 1024;
    let heads = 32;
    let hd = 128;

    let dtype = DType::F32;
    let q = Tensor::zeros((b, heads, seq, hd), dtype, device).unwrap();
    let k = Tensor::zeros((b, heads, seq, hd), dtype, device).unwrap();
    let v = Tensor::zeros((b, heads, seq, hd), dtype, device).unwrap();
    let m = Tensor::zeros((b, heads, seq, seq), dtype, device).unwrap();

    let flops = b * seq * heads * hd;

    let mut group = c.benchmark_group(device.bench_name("attention_fast"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                let _ = scaled_dot_product_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    0.3,
                    Some(black_box(&m)),
                    false,
                    seq,
                )
                .unwrap();
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark_fast(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        run_bench_fast(c, &device);
    }
}

criterion_group!(benches_naive, criterion_benchmark_naive);
criterion_group!(benches_fast, criterion_benchmark_fast);
