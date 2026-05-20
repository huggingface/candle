//! Microbench for the per-expert quantized matmul kernel
//! (`kernel_mul_mm_id`, exposed in candle-metal-kernels as
//! `call_quantized_matmul_mm_id`).
//!
//! This bench drives the kernel directly: there is no `QMatMul` integration
//! yet (the wrapper is meant to be invoked from a `CustomOp` by callers that
//! own their own MoE routing logic). The benchmark stages a quantized
//! per-expert weight stack `[E, N, K]`, an `f32` input `[M, K]`, an `i32`
//! routing table `[M, T]`, and dispatches the kernel in a tight loop.
//!
//! Shapes are picked to match Qwen3-MoE's FFN dims (N=2048, K=4096), with
//! M sweeping the decode (M=1) / small prefill (M=32) / chunked-prefill
//! (M=256) regimes. Dtype is fixed to Q4_K (the common MoE quantization).

use crate::benchmarks::BenchDeviceHandler;
use candle_core::Device;
use criterion::{criterion_group, Criterion};

#[cfg(feature = "metal")]
use crate::benchmarks::BenchDevice;
#[cfg(feature = "metal")]
use candle_core::{quantized, Tensor};
#[cfg(feature = "metal")]
use criterion::Throughput;
#[cfg(feature = "metal")]
use std::hint::black_box;
#[cfg(feature = "metal")]
use std::time::Instant;

#[cfg(feature = "metal")]
fn run_bench_metal(c: &mut Criterion, device: &Device, m: usize) {
    use candle_core::quantized::GgmlDType;

    let dtype = GgmlDType::Q4K;
    let num_experts = 128usize;
    let experts_per_tok = 8usize;
    let n = 2048usize;
    let k = 4096usize;

    // 1. Build quantized expert weights `[E, N, K]` on the CPU, then
    //    upload the raw quantized bytes to a Metal buffer.
    let weights_f32: Vec<f32> = (0..(num_experts * n * k))
        .map(|v| (v as f32) / ((num_experts * n * k) as f32))
        .collect();
    let weights_tensor =
        Tensor::from_slice(&weights_f32, (num_experts, n, k), &Device::Cpu).unwrap();
    let qtensor = quantized::QTensor::quantize(&weights_tensor, dtype).unwrap();
    let qbytes = qtensor.data().unwrap();

    let metal_device = match device {
        Device::Metal(d) => d.clone(),
        _ => return,
    };

    let weight_buf = metal_device.new_buffer_with_data(qbytes.as_ref()).unwrap();

    // Quantized strides (bytes). Layout is row-major over `[E, N, K]`.
    let type_size = dtype.type_size();
    let block_size = dtype.block_size();
    let row_bytes = (k / block_size) * type_size;
    let expert_bytes = n * row_bytes;
    let src0_shape = [num_experts, n, k];
    let src0_stride_bytes = [
        expert_bytes * num_experts,
        expert_bytes,
        row_bytes,
        type_size,
    ];
    // call_quantized_matmul_mm_id reads only nb01 (row stride) and nb02
    // (per-expert stride); the deeper strides are unused but we plumb a
    // 4-element stride array for symmetry with `mm_t`.
    let src0_stride = [
        src0_stride_bytes[0],
        src0_stride_bytes[1],
        src0_stride_bytes[2],
        src0_stride_bytes[3],
    ];

    // 2. Build an `f32` input `[M, K]` and upload.
    let input_f32: Vec<f32> = (0..(m * k))
        .map(|v| (v as f32) / ((m * k) as f32))
        .collect();
    let input_buf = metal_device.new_buffer_with_data(&input_f32).unwrap();
    let f32_size = core::mem::size_of::<f32>();
    let src1_shape = [m, k];
    let src1_stride = [k * f32_size, f32_size];

    // 3. Build a routing table `[M, T]` of `i32` expert ids in
    //    [0, num_experts).
    let ids_i32: Vec<i32> = (0..(m * experts_per_tok))
        .map(|v| ((v * 17) % num_experts) as i32)
        .collect();
    let ids_buf = metal_device.new_buffer_with_data(&ids_i32).unwrap();
    let i32_size = core::mem::size_of::<i32>();
    let ids_shape = [m, experts_per_tok];
    let ids_stride = [experts_per_tok * i32_size, i32_size];

    // 4. Allocate an output buffer `[M, T, N]` of `f32`.
    let out_elems = m * experts_per_tok * n;
    let out_buf = metal_device
        .new_buffer(out_elems, candle_core::DType::F32, "qmatmul_id_out")
        .unwrap();

    // FLOPs per dispatch: M tokens * T slots * 2 * N * K.
    let flops = (m as u64) * (experts_per_tok as u64) * 2 * (n as u64) * (k as u64);

    let bench_name = format!("qmatmul_id_q4k_n{}_k{}_m{}", n, k, m);
    let mut group = c.benchmark_group(device.bench_name(bench_name));
    group.sample_size(50);
    group.throughput(Throughput::Bytes(flops));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let encoder = metal_device.command_encoder().unwrap();
                candle_metal_kernels::call_quantized_matmul_mm_id(
                    metal_device.device(),
                    &encoder,
                    metal_device.kernels(),
                    dtype.into(),
                    black_box(&src0_shape),
                    black_box(&src0_stride),
                    black_box(&weight_buf),
                    black_box(&src1_shape),
                    black_box(&src1_stride),
                    black_box(&input_buf),
                    0,
                    black_box(&ids_shape),
                    black_box(&ids_stride),
                    black_box(&ids_buf),
                    0,
                    black_box(&[m, experts_per_tok, n]),
                    0,
                    black_box(&out_buf),
                )
                .unwrap();
                drop(encoder);
            }
            metal_device.wait_until_completed().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

#[cfg(not(feature = "metal"))]
fn run_bench_metal(_c: &mut Criterion, _device: &Device, _m: usize) {}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        if matches!(device, Device::Metal(_)) {
            for m in [1usize, 32, 256] {
                run_bench_metal(c, &device, m);
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
