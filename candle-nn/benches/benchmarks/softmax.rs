use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Tensor};
use candle_nn::ops::softmax_last_dim;
use criterion::Throughput;
use criterion::{criterion_group, Criterion};
use std::hint::black_box;
use std::time::Instant;

// Original optimized softmax algorithm for fair comparison
fn softmax_last_dim_original_optimized(xs: &Tensor) -> candle::Result<Tensor> {
    // This replicates the original algorithm using the same approach but with high-level ops
    // The key is maintaining the two-pass structure: max -> exp -> sum -> normalize
    let last_dim = xs.dims().len() - 1;
    
    // First pass: find max (equivalent to vec_reduce_max in original)
    let max_vals = xs.max_keepdim(last_dim)?;
    
    // Second pass: compute exp(x - max) (equivalent to the exp loop in original)
    let exp_vals = xs.broadcast_sub(&max_vals)?.exp()?;
    
    // Third pass: sum exponentials (equivalent to vec_reduce_sum in original)
    let sum_vals = exp_vals.sum_keepdim(last_dim)?;
    
    // Final pass: normalize (equivalent to the division loop in original)
    let result = exp_vals.broadcast_div(&sum_vals)?;
    
    Ok(result)
}

fn run_online(input: &Tensor) {
    let _ = softmax_last_dim(input).unwrap();
}

fn run_original_optimized(input: &Tensor) {
    let _ = softmax_last_dim_original_optimized(input).unwrap();
}

const B: usize = 1;
const M: usize = 1024;
const K: usize = 1024;

fn run_softmax_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let elements = B * M * K;

    let input = Tensor::rand(-1000.0f32, 1000.0f32, (B, M, K), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let flops = elements * dtype.size_in_bytes();
    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    
    // Online softmax benchmark
    let input_online = input.clone();
    group.bench_function("online", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_online(black_box(&input_online));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    
    // Original optimized softmax benchmark
    let input_original = input.clone();
    group.bench_function("original_optimized", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_original_optimized(black_box(&input_original));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = BenchDeviceHandler::new().unwrap();
    for d in device.devices {
        run_softmax_benchmark(c, &d, DType::F32, "softmax_f32");
        run_softmax_benchmark(c, &d, DType::BF16, "softmax_bf16");
        run_softmax_benchmark(c, &d, DType::F16, "softmax_f16");
    }
}

// Additional benchmark for different tensor sizes
fn run_softmax_size_benchmark(c: &mut Criterion, device: &Device, b_size: usize, m_size: usize, k_size: usize) {
    let elements = b_size * m_size * k_size;
    let input = Tensor::rand(-1000.0f32, 1000.0f32, (b_size, m_size, k_size), device).unwrap();
    
    let flops = elements * std::mem::size_of::<f32>();
    let mut group = c.benchmark_group(format!("cpu_softmax_size_{}_{}_{}", b_size, m_size, k_size));
    group.throughput(Throughput::Bytes(flops as u64));
    
    // Online softmax
    let input_online = input.clone();
    group.bench_function("online", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_online(black_box(&input_online));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    
    // Original optimized softmax
    let input_original = input.clone();
    group.bench_function("original_optimized", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_original_optimized(black_box(&input_original));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    
    group.finish();
}

#[cfg(test)]
mod size_benchmarks {
    use super::*;
    
    #[test]
    fn benchmark_different_sizes() {
        let mut c = Criterion::default().configure_from_args();
        let device = BenchDeviceHandler::new().unwrap();
        
        // Test different tensor sizes
        let sizes = [
            (1, 256, 256),    // Small
            (1, 512, 512),    // Medium  
            (1, 1024, 1024),  // Large (default)
            (2, 512, 512),    // Batch
            (1, 2048, 512),   // Wide
        ];
        
        for (b, m, k) in sizes.iter() {
            run_softmax_size_benchmark(&mut c, &device.devices[0], *b, *m, *k);
        }
    }
}

criterion_group!(benches, criterion_benchmark);
