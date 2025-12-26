use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Tensor};
use candle_nn::ops::softmax_last_dim;
use rayon::prelude::*;
use criterion::Throughput;
use criterion::{criterion_group, Criterion};
use std::hint::black_box;
use std::time::Instant;

// Original optimized softmax algorithm - EXACT replica with vec_reduce_max/vec_reduce_sum
fn softmax_last_dim_original_optimized(xs: &Tensor) -> candle::Result<Tensor> {
    // Create a custom op that implements the EXACT original pattern
    struct OriginalSoftmaxLastDim;
    
    impl candle::CustomOp1 for OriginalSoftmaxLastDim {
        fn name(&self) -> &'static str {
            "original-softmax-last-dim"
        }
        
        fn cpu_fwd(&self, storage: &candle::CpuStorage, layout: &candle::Layout) -> candle::Result<(candle::CpuStorage, candle::Shape)> {
            fn softmax<T: candle::WithDType + num_traits::Float>(
                src: &[T],
                layout: &candle::Layout,
            ) -> candle::Result<(candle::CpuStorage, candle::Shape)> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => &src[o1..o2],
                };
                let el_count = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let mut dst = vec![T::zero(); el_count];
                src.par_chunks(dim_m1)
                    .zip(dst.par_chunks_mut(dim_m1))
                    .for_each(|(src, dst): (&[T], &mut [T])| {
                        let mut max = T::neg_infinity();
                        unsafe { T::vec_reduce_max(src.as_ptr(), &mut max, dim_m1) };
                        for (s, d) in src.iter().zip(dst.iter_mut()) {
                            *d = (*s - max).exp();
                        }
                        let mut sum_exp = T::zero();
                        unsafe { T::vec_reduce_sum(dst.as_ptr(), &mut sum_exp, dim_m1) };
                        for d in dst.iter_mut() {
                            *d /= sum_exp;
                        }
                    });
                let storage = candle::WithDType::to_cpu_storage_owned(dst);
                Ok((storage, candle::Shape::from_dims(dims)))
            }

            match storage {
                candle::CpuStorage::BF16(slice) => softmax::<half::bf16>(slice, layout),
                candle::CpuStorage::F16(slice) => softmax::<half::f16>(slice, layout),
                candle::CpuStorage::F32(slice) => softmax::<f32>(slice, layout),
                candle::CpuStorage::F64(slice) => softmax::<f64>(slice, layout),
                _ => candle::bail!("unsupported dtype for softmax {:?}", storage),
            }
        }
    }
    
    xs.apply_op1_no_bwd(&OriginalSoftmaxLastDim)
}

fn run_online(input: &Tensor) {
    let _ = softmax_last_dim(input).unwrap();
}

fn run_original_optimized(input: &Tensor) {
    let _ = softmax_last_dim_original_optimized(input).unwrap();
}

#[test]
fn test_original_optimized_correctness() {
    let device = Device::Cpu;
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device).unwrap();
    
    let online_result = softmax_last_dim(&tensor).unwrap();
    let original_result = softmax_last_dim_original_optimized(&tensor).unwrap();
    
    // They should be very close
    let diff = (online_result - original_result).unwrap().abs().sum_all().unwrap().to_vec0::<f32>().unwrap();
    assert!(diff < 1e-6, "Results differ too much: {}", diff);
    
    println!("✅ Original implementation correctness verified - difference: {}", diff);
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
    for d in &device.devices {
        run_softmax_benchmark(c, d, DType::F32, "softmax_f32");
        run_softmax_benchmark(c, d, DType::BF16, "softmax_bf16");
        run_softmax_benchmark(c, d, DType::F16, "softmax_f16");
    }
    
    // Additional shape benchmarks
    run_shape_benchmarks(c, &device.devices[0]);
}

fn run_shape_benchmarks(c: &mut Criterion, device: &Device) {
    let shapes = [
        // Standard square (baseline)
        (1, 1024, 1024, "square_standard"),
        
        // Long and thin tensors
        (1, 65536, 16, "long_thin_64k"),
        
        // Very long single dimension
        (1, 1, 65536, "very_long_1d"),
        
        // Wide and short tensors
        (1, 64, 16384, "wide_short_16k"),
        
        // Multiple batches
        (8, 512, 512, "batch_8x512x512"),

        // Large last dimension
        (1, 128, 8192, "large_last_8k"),
    ];
    
    for (b, m, k, name) in shapes.iter() {
        let elements = b * m * k;
        let input = Tensor::rand(-1000.0f32, 1000.0f32, (*b, *m, *k), device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        
        let flops = elements * std::mem::size_of::<f32>();
        let mut group = c.benchmark_group(format!("shape_{}", name));
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
