use crate::utils::EncoderProvider;
use crate::{
    set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

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
            Output::with_offset(dst, dst_offset),
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
            Output::with_offset(dst, dst_offset),
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

    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Indirect / per-expert quantized matmul, as used by MoE FFNs.
///
/// Mirrors `kernel_mul_mm_id` in `metal_src/quantized.metal` (a single-pass
/// adaptation of the llama.cpp MoE kernel). The weight tensor `src0s` is a
/// stack of `num_experts` quantized matrices of shape `[n, k]`. The input
/// `src1` is a dense `[num_tokens, k]` matrix in `f32`. `ids` selects which
/// experts each token routes to, shape `[num_tokens, experts_per_token]` in
/// `i32`. The output `dst` is `f32` of shape
/// `[num_tokens, experts_per_token, n]` laid out as
/// `dst[t, e, j] = dst[t * experts_per_token * n + e * n + j]`, i.e. each
/// `(token, slot)` pair writes one row of size `n`.
///
/// This is the single-pass variant; llama.cpp's `_id_map0` pre-pass is not
/// vendored here. Callers that need to integrate with `QMatmul` should drive
/// this from a `CustomOp`; auto-dispatch through `forward_via_f16` is
/// intentionally out of scope (see ivarflakstad's review of candle #3444).
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_id(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    // src0s: [num_experts, n, k] quantized
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0s: &Buffer,
    // src1: [num_tokens, k] f32
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    // ids: [num_tokens, experts_per_token] i32
    ids_shape: &[usize],
    ids_stride: &[usize],
    ids: &Buffer,
    ids_offset: usize,
    // dst: [num_tokens, experts_per_token, n] f32
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    if src0_shape.len() < 3 {
        return Err(MetalKernelError::InvalidInput(format!(
            "qmatmul_id: src0 must have rank >= 3 (got {:?})",
            src0_shape
        )));
    }
    if src1_shape.len() < 2 {
        return Err(MetalKernelError::InvalidInput(format!(
            "qmatmul_id: src1 must have rank >= 2 (got {:?})",
            src1_shape
        )));
    }
    if ids_shape.len() != 2 {
        return Err(MetalKernelError::InvalidInput(format!(
            "qmatmul_id: ids must have rank == 2 (got {:?})",
            ids_shape
        )));
    }

    // src0s shape: [..., ne02, ne01, ne00] = [..., num_experts, n, k]
    let r = src0_shape.len();
    let ne00 = src0_shape[r - 1] as i64;
    let ne01 = src0_shape[r - 2] as i64;
    let ne02 = src0_shape[r - 3] as i64;

    let sr = src0_stride.len();
    let nb01 = src0_stride[sr - 2] as i64;
    let nb02 = src0_stride[sr - 3] as i64;

    // src1 shape: [..., num_tokens, k]. The kernel reads it as 4D
    // [ne13, ne12, ne11, ne10] with byte strides nb1x. For the common
    // single-batch case ne13 = ne12 = 1, ne11 = num_tokens, ne10 = k.
    let r1 = src1_shape.len();
    let ne11 = if r1 >= 2 {
        src1_shape[r1 - 2] as i64
    } else {
        1
    };
    let ne12 = if r1 >= 3 {
        src1_shape[r1 - 3] as i64
    } else {
        1
    };
    let ne13 = if r1 >= 4 {
        src1_shape[r1 - 4] as i64
    } else {
        1
    };

    let s1 = src1_stride.len();
    let nb10 = src1_stride[s1 - 1] as i64;
    let nb11 = if s1 >= 2 {
        src1_stride[s1 - 2] as i64
    } else {
        0
    };
    let nb12 = if s1 >= 3 {
        src1_stride[s1 - 3] as i64
    } else {
        0
    };

    // ids shape: [nei1, nei0] = [num_tokens, experts_per_token].
    let nei0 = ids_shape[1] as i64;
    let nei1 = ids_shape[0] as i64;
    let nbi1 = ids_stride[0] as i64;

    // dst shape: trailing dims are [..., ne1, ne0]. ne0 is n (cols per
    // expert output); ne1 is the total slot count (num_tokens *
    // experts_per_token) when laid out flat. The MSL only reads ne0, ne1,
    // and nb1; the kernel itself addresses rows through `rowids` and
    // strides through `ne0` / `ne0*ne1`.
    let dr = dst_shape.len();
    let ne0 = dst_shape[dr - 1] as i64;
    let ne1 = (nei0 * nei1) as i64;
    let nb1 = (ne0 as usize * core::mem::size_of::<f32>()) as i64;

    let thread_groups_count = MTLSize {
        width: divide(ne1 as usize, 32),
        height: divide(ne0 as usize, 64),
        depth: ne02 as usize,
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
        GgmlDType::BF16 => Err(MetalKernelError::UnsupportedDTypeForOp(
            "BF16",
            "qmatmul_id",
        ))?,
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp(
            "Q8_1",
            "qmatmul_id",
        ))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "qmatmul_id"))?,
    };

    // Silence unused-warnings for shape fields the kernel does not read.
    let _ = ne01;

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            src0s,
            (src1, src1_offset),
            Output::with_offset(dst, dst_offset),
            (ids, ids_offset),
            nei0,
            nei1,
            nbi1,
            ne00,
            ne02,
            nb01,
            nb02,
            ne11,
            ne12,
            ne13,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            nb1
        )
    );

    // 8192 bytes for the simdgroup tile buffer (matches kernel_mul_mm), plus
    // a ushort2 per (token, slot) pair used by the rowids fan-out at the top
    // of kernel_mul_mm_id.
    let shared_bytes = 8192 + (nei0 as usize) * (nei1 as usize) * core::mem::size_of::<u32>();
    encoder.set_threadgroup_memory_length(0, shared_bytes);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

fn divide(m: usize, b: usize) -> usize {
    m.div_ceil(b)
}
