//! Microbench for the per-expert quantized matmul kernel
//! (`kernel_mul_mm_id`, exposed in candle-metal-kernels as
//! `call_quantized_matmul_mm_id`), plus a baseline that drives the existing
//! `QMatMul::forward` path at the equivalent per-expert work shape.
//!
//! The fused bench (`qmatmul_id_q4k_*`) stages a quantized per-expert weight
//! stack `[E, N, K]`, an `f32` input `[M, K]`, an `i32` routing table
//! `[M, T]`, and dispatches `kernel_mul_mm_id` in a tight loop. Shapes match
//! Qwen3-MoE's FFN dims (N=2048, K=4096); M sweeps decode (1), small
//! prefill (32), and chunked prefill (256). Dtype is Q4_K (the common MoE
//! quantization).
//!
//! The baseline bench (`qmatmul_mm_t_baseline_q4k_*`) measures a single
//! expert's worth of work through `QMatMul::forward` at M equal to the
//! average tokens-per-expert in the fused case (T=8, E=128 -> 0, 2, 16 for
//! the three M values). The naive 128-dispatch alternative would pay this
//! cost once per expert; comparing `fused_time` to `baseline_time * E`
//! shows when the fused kernel actually wins.
//!
//! The vendored `kernel_mul_mm_id` at `quantized.metal:7301` has an
//! unparallelized rowids fan-out (TODO in the upstream source): every
//! thread of every threadgroup independently scans the full `M*T` ids
//! buffer. This is visible as the M=256 regression in the fused bench.
//!
//! Units: the throughput field carries FLOPs (`2*M*N*K` per matmul, times
//! T for the fused case). Criterion prints it as "Gelem/s" but the value
//! is GFLOPs/s.

use crate::benchmarks::BenchDeviceHandler;
use candle_core::Device;
use criterion::{criterion_group, Criterion};

#[cfg(feature = "metal")]
use crate::benchmarks::BenchDevice;
#[cfg(feature = "metal")]
use candle_core::{
    quantized::{self, QMatMul},
    Module, Tensor,
};
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
    // Pass FLOPs as `Elements`; Criterion will print "Gelem/s" but the value
    // is GFLOPs/s.
    group.throughput(Throughput::Elements(flops));
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

#[cfg(feature = "metal")]
fn run_bench_mm_t_baseline(c: &mut Criterion, device: &Device, m_per_expert: usize) {
    use candle_core::quantized::GgmlDType;

    let dtype = GgmlDType::Q4K;
    let n = 2048usize;
    let k = 4096usize;

    let lhs_data: Vec<f32> = (0..(m_per_expert * k))
        .map(|v| (v as f32) / ((m_per_expert * k) as f32))
        .collect();
    let rhs_data: Vec<f32> = (0..(n * k)).map(|v| (v as f32) / ((n * k) as f32)).collect();

    let lhs = Tensor::from_slice(&lhs_data, (m_per_expert, k), device).unwrap();
    let rhs = Tensor::from_slice(&rhs_data, (k, n), device).unwrap();
    let qtensor = quantized::QTensor::quantize(&rhs.t().unwrap(), dtype).unwrap();
    let matmul = QMatMul::from_qtensor(qtensor).unwrap();

    let flops = (m_per_expert as u64) * 2 * (n as u64) * (k as u64);

    let bench_name = format!("qmatmul_mm_t_baseline_q4k_n{}_k{}_m{}", n, k, m_per_expert);
    let mut group = c.benchmark_group(device.bench_name(bench_name));
    group.sample_size(50);
    group.throughput(Throughput::Elements(flops));
    let lhs = black_box(lhs);
    let matmul = black_box(matmul);
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                matmul.forward(&lhs).unwrap();
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

#[cfg(not(feature = "metal"))]
fn run_bench_mm_t_baseline(_c: &mut Criterion, _device: &Device, _m_per_expert: usize) {}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        if matches!(device, Device::Metal(_)) {
            for m in [1usize, 32, 256] {
                run_bench_metal(c, &device, m);
            }
            // Baseline through the existing `QMatMul::forward` path, one
            // dispatch per expert. The pairings are: M=1 / decode -> only
            // T=8 experts hit (M_per_expert=1); M=32 -> 256/128 = 2 average;
            // M=256 -> 2048/128 = 16 average. The naive 128-dispatch cost
            // is roughly `baseline_time(m_per_expert) * num_experts_hit`.
            for m_per_expert in [1usize, 2, 16] {
                run_bench_mm_t_baseline(c, &device, m_per_expert);
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
