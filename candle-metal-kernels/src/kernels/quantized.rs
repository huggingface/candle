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

fn divide(m: usize, b: usize) -> usize {
    m.div_ceil(b)
}

/// Indexed quantized matrix multiplication for MoE (Mixture of Experts)
///
/// This is essentially a "gather_qmm" operation that performs:
/// - For each row i: output[i] = input[i] @ dequantize(weights[ids[i]]).T
///
/// The dequantization happens fused inside the kernel, not as a separate pass.
///
/// # Arguments
/// * `src0` - Expert weights buffer (quantized) [num_experts, n, k]
/// * `src1` - Input activations buffer [batch, seq, k]
/// * `dst` - Output buffer [batch, seq, n]
/// * `ids` - Expert indices buffer [batch, seq] (int32)
/// * `nei0` - Number of indices dimension 0 (typically num_experts_per_token)
/// * `nei1` - Number of indices dimension 1 (typically batch * seq)
/// * `nbi1` - Stride for indices dimension 1
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_id_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    // Index dimensions
    (nei0, nei1): (usize, usize),
    nbi1: usize,
    // src0 (weights) shape and stride - [num_experts, n, k]
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    // src1 (input) shape and stride
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    // dst (output) shape and stride
    dst_shape: &[usize],
    dst_stride: &[usize],
    dst_offset: usize,
    dst: &Buffer,
    // ids (expert indices)
    ids: &Buffer,
) -> Result<(), MetalKernelError> {
    // src0 shape: [num_experts, n, k] - weights
    // In GGML convention, shapes are stored in reverse (k, n, num_experts, 1)
    let ne00 = src0_shape[src0_shape.len() - 1] as i64; // k (hidden dim)
    let ne02 = src0_shape[src0_shape.len() - 3] as i64; // num_experts

    let nb01 = src0_stride[src0_stride.len() - 2] as u64; // stride for n dimension
    let nb02 = src0_stride[src0_stride.len() - 3] as u64; // stride for expert dimension

    // src1 shape: [batch, seq, k] - input
    let ne11 = src1_shape[src1_shape.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64;
    let ne13 = if src1_shape.len() >= 4 {
        src1_shape[src1_shape.len() - 4] as i64
    } else {
        1
    };

    let nb10 = src1_stride[src1_stride.len() - 1] as u64;
    let nb11 = src1_stride[src1_stride.len() - 2] as u64;
    let nb12 = src1_stride[src1_stride.len() - 3] as u64;

    // dst shape: [batch, seq, n] - output
    let ne0 = dst_shape[dst_shape.len() - 1] as i64; // n (output features)
    let ne1 = dst_shape[dst_shape.len() - 2] as i64; // seq

    let nb1 = dst_stride[dst_stride.len() - 2] as u64;

    // Thread group configuration for indexed matmul
    // The kernel iterates over (token, expert_slot) pairs that selected each expert.
    // Maximum possible pairs per expert is nei0 * nei1 (all tokens selected this expert).
    // - width: columns = max (token, expert_slot) pairs = nei0 * nei1
    // - height: rows = output features (n dimension)
    // - depth: number of experts (each thread group z handles one expert)
    let thread_groups_count = MTLSize {
        width: divide(nei0 * nei1, 32),                        // max column blocks
        height: divide(src0_shape[src0_shape.len() - 2], 64),  // n dimension
        depth: ne02 as usize,                                  // num_experts
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };

    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mm_id_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mm_id_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mm_id_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mm_id_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mm_id_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mm_id_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mm_id_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mm_id_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mm_id_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mm_id_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mm_id_f16_f32",
        GgmlDType::F32 => "kernel_mul_mm_id_f32_f32",
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp(
            "Q8_1",
            "qmatmul_id",
        ))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp(
            "Q8K",
            "qmatmul_id",
        ))?,
        GgmlDType::BF16 => Err(MetalKernelError::UnsupportedDTypeForOp(
            "BF16",
            "qmatmul_id",
        ))?,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Convert index params
    let nei0 = nei0 as i64;
    let nei1 = nei1 as i64;
    let nbi1 = nbi1 as u64;

    set_params!(
        encoder,
        (
            src0,                    // buffer 0: expert weights
            (src1, src1_offset),     // buffer 1: input activations
            (dst, dst_offset),       // buffer 2: output
            ids,                     // buffer 3: expert indices
            nei0,                    // buffer 4: index dimension 0
            nei1,                    // buffer 5: index dimension 1
            nbi1,                    // buffer 6: index stride
            ne00,                    // buffer 7: k (hidden dim)
            ne02,                    // buffer 8: num_experts
            nb01,                    // buffer 9: weight stride[1]
            nb02,                    // buffer 10: weight stride[2]
            ne11,                    // buffer 11: src1 shape[1]
            ne12,                    // buffer 12: src1 shape[2]
            ne13,                    // buffer 13: src1 shape[3]
            nb10,                    // buffer 14: src1 stride[0]
            nb11,                    // buffer 15: src1 stride[1]
            nb12,                    // buffer 16: src1 stride[2]
            ne0,                     // buffer 17: output features (n)
            ne1,                     // buffer 18: output seq
            nb1                      // buffer 19: output stride
        )
    );

    encoder.use_resource(src0, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(ids, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    // Threadgroup memory: 8192 for shared_memory + space for rowids
    // The kernel uses: threadgroup ushort2 * rowids = (threadgroup ushort2 *)(shared_memory + 8192)
    // Maximum rowids needed: nei0 * nei1 * sizeof(ushort2) = nei0 * nei1 * 4
    let rowids_size = (nei0 as usize) * (nei1 as usize) * 4;
    let threadgroup_mem_size = 8192 + rowids_size.max(8192);
    encoder.set_threadgroup_memory_length(0, threadgroup_mem_size);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}
