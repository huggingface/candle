use crate::utils::EncoderProvider;
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

#[derive(Debug, Clone, Copy)]
pub enum GgmlDType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    F16,
    F32,
    BF16,
}

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;

    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;

    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    let (nth0, nth1, align) = match dtype {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q8_1 => {
            let nth0 = 8;
            let nth1 = 8;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::Q2K => {
            // Fixing a bug in Metal for GGML
            // https://github.com/ggerganov/llama.cpp/blob/b8109bc0139f15a5b321909f47510b89dca47ffc/ggml-metal.m#L1576
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q4K => {
            let nth0 = 4;
            let nth1 = 8;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q3K | GgmlDType::Q5K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q6K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 2;
            (nth0, nth1, align)
        }
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K => {
            // Original implem uses rows
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::F32 => {
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
    };
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mv_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_q8_0_f32",
        GgmlDType::Q8_1 => "kernel_mul_mv_q8_1_f32",
        GgmlDType::Q2K => "kernel_mul_mv_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mv_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_q6_K_f32",
        GgmlDType::Q8K => "kernel_mul_mv_q8_K_f32",
        GgmlDType::F16 => "kernel_mul_mv_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mv_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mv_f32_f32",
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            (dst, dst_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(lhs, MTLResourceUsage::Read);
    encoder.use_resource(rhs, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// - src0 is usually weight
/// - src1 is usually xs
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = src0_shape[src0_shape.len() - 1] as i64;
    let ne01 = src0_shape[src0_shape.len() - 2] as i64;
    let ne02 = src0_shape[src0_shape.len() - 3] as i64;
    let ne03 = src0_shape[src0_shape.len() - 4] as i64;

    let nb01 = src0_stride[src0_stride.len() - 2] as i64;
    let nb02 = src0_stride[src0_stride.len() - 3] as i64;
    let nb03 = src0_stride[src0_stride.len() - 4] as i64;

    let ne11 = src1_shape[src1_shape.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64;
    let ne13 = src1_shape[src1_shape.len() - 4] as i64;

    let nb10 = src1_stride[src1_stride.len() - 1] as i64;
    let nb11 = src1_stride[src1_stride.len() - 2] as i64;
    let nb12 = src1_stride[src1_stride.len() - 3] as i64;
    let nb13 = src1_stride[src1_stride.len() - 4] as i64;

    let ne0 = dst_shape[dst_shape.len() - 1] as i64;
    let ne1 = dst_shape[dst_shape.len() - 2] as i64;
    let r2 = (ne12 / ne02) as u32;
    let r3 = (ne13 / ne03) as u32;

    let thread_groups_count = MTLSize {
        width: divide(ne11 as usize, 32),
        height: divide(ne01 as usize, 64),
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mm_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mm_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mm_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mm_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mm_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mm_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mm_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mm_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mm_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mm_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mm_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mm_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mm_f32_f32",
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "qmatmul"))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "qmatmul"))?,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            src0,
            (src1, src1_offset),
            (dst, dst_offset),
            ne00,
            ne02,
            nb01,
            nb02,
            nb03,
            ne12,
            nb10,
            nb11,
            nb12,
            nb13,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Dequantize a quantized tensor to F32 using Metal GPU kernels.
///
/// Uses `kernel_get_rows_*` to dequantize all rows of a quantized weight tensor
/// directly on the GPU, avoiding the CPU roundtrip of `QMetalStorage::dequantize()`.
///
/// Parameters:
/// - `n_rows`: number of rows in the weight tensor
/// - `n_cols`: number of columns (elements per row, must be multiple of block_size)
/// - `src`: quantized source buffer
/// - `dst`: pre-allocated F32 output buffer (n_rows * n_cols * 4 bytes)
/// - `indices`: buffer containing sequential i32 indices [0, 1, ..., n_rows-1]
#[allow(clippy::too_many_arguments)]
pub fn call_dequantize_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    n_rows: usize,
    n_cols: usize,
    src: &Buffer,
    dst: &Buffer,
    indices: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_get_rows_q4_0",
        GgmlDType::Q4_1 => "kernel_get_rows_q4_1",
        GgmlDType::Q5_0 => "kernel_get_rows_q5_0",
        GgmlDType::Q5_1 => "kernel_get_rows_q5_1",
        GgmlDType::Q8_0 => "kernel_get_rows_q8_0",
        GgmlDType::Q2K => "kernel_get_rows_q2_K",
        GgmlDType::Q3K => "kernel_get_rows_q3_K",
        GgmlDType::Q4K => "kernel_get_rows_q4_K",
        GgmlDType::Q5K => "kernel_get_rows_q5_K",
        GgmlDType::Q6K => "kernel_get_rows_q6_K",
        GgmlDType::F16 => "kernel_get_rows_f16",
        GgmlDType::BF16 => "kernel_get_rows_bf16",
        GgmlDType::F32 => "kernel_get_rows_f32",
        GgmlDType::Q8_1 => {
            return Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "dequantize"))
        }
        GgmlDType::Q8K => {
            return Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "dequantize"))
        }
    };

    // Compute byte strides for the kernel
    let block_size = dtype_block_size(dtype);
    let type_size = dtype_type_size(dtype);
    // nb01 = byte stride per row in quantized source
    let nb01: u64 = (n_cols / block_size * type_size) as u64;
    // nb02 = byte stride per batch (single batch = all rows)
    let nb02: u64 = n_rows as u64 * nb01;
    // ne00 = number of output elements per row
    let ne00: i64 = n_cols as i64;
    // ne10 = number of indices (= number of rows)
    let ne10: i64 = n_rows as i64;
    // nb10 = stride between indices in bytes (sizeof(i32) = 4)
    let nb10: u64 = 4;
    // nb11 = batch stride for indices (single batch)
    let nb11: u64 = n_rows as u64 * 4;
    // nb1 = output row stride in bytes
    let nb1: u64 = n_cols as u64 * 4;
    // nb2 = output batch stride in bytes
    let nb2: u64 = n_rows as u64 * n_cols as u64 * 4;

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (src, indices, dst, ne00, nb01, nb02, ne10, nb10, nb11, nb1, nb2)
    );
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(indices, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    let thread_groups_count = MTLSize {
        width: n_rows,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Block size for each GGML dtype.
fn dtype_block_size(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::F32 => 1,
        GgmlDType::F16 | GgmlDType::BF16 => 1,
        GgmlDType::Q4_0 => 32,
        GgmlDType::Q4_1 => 32,
        GgmlDType::Q5_0 => 32,
        GgmlDType::Q5_1 => 32,
        GgmlDType::Q8_0 => 32,
        GgmlDType::Q8_1 => 32,
        GgmlDType::Q2K => 256,
        GgmlDType::Q3K => 256,
        GgmlDType::Q4K => 256,
        GgmlDType::Q5K => 256,
        GgmlDType::Q6K => 256,
        GgmlDType::Q8K => 256,
    }
}

/// Type size in bytes for each GGML dtype block.
fn dtype_type_size(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::F32 => 4,
        GgmlDType::F16 | GgmlDType::BF16 => 2,
        GgmlDType::Q4_0 => 18,   // 2 (delta) + 16 (qs)
        GgmlDType::Q4_1 => 20,   // 2 (delta) + 2 (min) + 16 (qs)
        GgmlDType::Q5_0 => 22,   // 2 (delta) + 4 (qh) + 16 (qs)
        GgmlDType::Q5_1 => 24,   // 2 (delta) + 2 (min) + 4 (qh) + 16 (qs)
        GgmlDType::Q8_0 => 34,   // 2 (delta) + 32 (qs)
        GgmlDType::Q8_1 => 40,   // 4 (delta) + 4 (sum) + 32 (qs)
        GgmlDType::Q2K => 84,    // block_q2_K
        GgmlDType::Q3K => 110,   // block_q3_K
        GgmlDType::Q4K => 144,   // block_q4_K
        GgmlDType::Q5K => 176,   // block_q5_K
        GgmlDType::Q6K => 210,   // block_q6_K
        GgmlDType::Q8K => 292,   // block_q8_K
    }
}

fn divide(m: usize, b: usize) -> usize {
    m.div_ceil(b)
}
