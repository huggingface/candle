use crate::utils::EncoderProvider;
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use objc2_metal::{MTLDevice, MTLSize};

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
    debug_group!(encoder, "qmm_mv {name} B={b} M={m} K={k} N={n}");

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
    debug_group!(encoder, "qmm_mm {name} M={ne11} K={ne00} N={ne01}");

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

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_get_rows(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    hidden_size: usize,
    row_stride: usize,
    ids_len: usize,
    src: &Buffer,
    ids: &Buffer,
    ids_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_row_stride = hidden_size * core::mem::size_of::<f32>();
    let name = match dtype {
        GgmlDType::F32 => "kernel_get_rows_f32",
        GgmlDType::F16 => "kernel_get_rows_f16",
        GgmlDType::BF16 => "kernel_get_rows_bf16",
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
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "get_rows"))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "get_rows"))?,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "qget_rows {name} ids={ids_len} hidden={hidden_size}"
    );

    let thread_groups_count = MTLSize {
        width: ids_len,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };

    set_params!(
        encoder,
        (
            src,
            (ids, ids_offset),
            Output::new(dst),
            hidden_size as i64,
            row_stride as u64,
            0u64,
            ids_len as i64,
            core::mem::size_of::<u32>() as u64,
            0u64,
            dst_row_stride as u64,
            0u64
        )
    );

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Indexed/routed matmul for MoE expert dispatch (ggml's `mul_mat_id`).
///
/// `src0` holds all experts' weights stacked on their leading dim
/// (`[n_expert, n_out, n_in]`); `ids` is the routing table
/// (`[n_tokens, n_expert_used]`, dtype i32) mapping each token's selected
/// experts to rows of `src0`; `dst` is `[n_expert_used * n_tokens, n_out]`,
/// one row per (token, selected-expert) pair, in `ids` row-major order. This
/// does *not* apply top-k routing weights or sum across experts -- callers
/// must do that in a second pass over `dst`.
///
/// Dispatch shape and threadgroup memory sizing are load-bearing constants
/// mirrored from ggml's Metal backend (`ggml_metal_encode_node`, the
/// `GGML_OP_MUL_MAT_ID` mm case circa llama.cpp commit b86f6007, the last
/// revision before the kernel was refactored to a 3-pass map0/mm/map1
/// pipeline) rather than derived from the kernel body: the kernel's own
/// `shared_memory + 8192` offset only makes sense if the host allocates at
/// least that many bytes plus room for the `rowids` scratch array, and the
/// per-expert dispatch depth isn't recoverable from the kernel signature at
/// all.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_id(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    src0_offset: usize,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    ids_shape: &[usize],
    ids_stride: &[usize],
    ids: &Buffer,
    ids_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse, same convention as call_quantized_matmul_mm_t.
    let ne00 = src0_shape[src0_shape.len() - 1] as i64; // n_in (contraction dim)
    let ne01 = src0_shape[src0_shape.len() - 2] as i64; // n_out
    let ne02 = src0_shape[src0_shape.len() - 3] as i64; // n_expert

    let nb01 = src0_stride[src0_stride.len() - 2] as i64;
    let nb02 = src0_stride[src0_stride.len() - 3] as i64;

    let nb10 = src1_stride[src1_stride.len() - 1] as i64;
    let ne11 = src1_shape[src1_shape.len() - 2] as i64; // n_expert_used (bcast) or 1
    let nb11 = src1_stride[src1_stride.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64; // n_tokens
    let nb12 = src1_stride[src1_stride.len() - 3] as i64;
    let ne13 = 1i64;

    let nei0 = ids_shape[ids_shape.len() - 1] as i64; // n_expert_used
    let nei1 = ids_shape[ids_shape.len() - 2] as i64; // n_tokens
    let nbi1 = ids_stride[ids_stride.len() - 2] as i64;

    let ne0 = dst_shape[dst_shape.len() - 1] as i64; // n_out
                                                     // dst is [n_tokens, n_expert_used, n_out] row-major; the kernel only ever
                                                     // uses ne1 to derive the per-token stride ne0*ne1, so it must equal
                                                     // n_expert_used (nei0) by construction, not the row count nei0*nei1 --
                                                     // reading it back out of dst_shape would silently corrupt the token
                                                     // stride if a caller ever passed a differently-shaped dst.
    let ne1 = nei0;
    let nb1 = ne0 * 4; // unused by the kernel; kept for ABI parity

    let thread_groups_count = MTLSize {
        width: divide(nei1 as usize, 32),
        height: divide(ne01 as usize, 64),
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
        GgmlDType::BF16 => "kernel_mul_mm_id_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mm_id_f32_f32",
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp(
            "Q8_1",
            "qmatmul_id",
        ))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "qmatmul_id"))?,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "qmm_mm_id {name} n_tokens={nei1} K={ne00} N={ne01} n_expert={ne02}"
    );

    set_params!(
        encoder,
        (
            (src0, src0_offset),
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

    // ggml's GGML_PAD(8192 + dst_rows*sizeof(ushort2), 16): 8192 bytes for the
    // mm tile scratch, then room for one ushort2 (4 bytes) per (token,
    // expert-slot) pair the `rowids` scan can find for this threadgroup's
    // expert -- worst case every row in the tile, i.e. nei0*nei1 total.
    let rowids_bytes = (nei0 * nei1 * 4) as usize;
    let smem = (8192 + rowids_bytes).div_ceil(16) * 16;
    // ggml's own host code asserts this same bound (dst_rows <=
    // dst_rows_max, derived from maxThreadgroupMemoryLength) before
    // dispatching -- unbounded batch*n_expert_used could otherwise exceed
    // the device's threadgroup memory budget and fail or misbehave on-device
    // rather than surfacing a clear error here.
    let max_smem = device.as_ref().maxThreadgroupMemoryLength() as usize;
    if smem > max_smem {
        return Err(MetalKernelError::InvalidInput(format!(
            "qmm_mm_id: required threadgroup memory ({smem} bytes, from nei0={nei0} * nei1={nei1}) exceeds this device's max ({max_smem} bytes)"
        )));
    }
    encoder.set_threadgroup_memory_length(0, smem);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Indexed/routed matmul for MoE expert dispatch, matrix-*vector* variant
/// (ggml's `mul_mat_id`, batch-1 case) -- companion to
/// `call_quantized_matmul_mm_id`, same input-shape contract (`src0`/`ids`/
/// `dst` layouts), so callers can dispatch either interchangeably based on
/// how many tokens a call covers. Intended for decode (`n_tokens == 1`),
/// where `mm_id`'s matrix-matrix kernel degenerates into many tiny tile
/// dispatches per expert.
///
/// Ground truth (dispatch grid, per-dtype `nth0`/`nth1`/width-divisor
/// tuning, buffer/scalar binding order, and the absence of any threadgroup
/// memory requirement) is reverse-derived from ggml's Metal encode path at
/// `llama.cpp` commit `f3f65429` -- independently confirmed to be the exact
/// commit this crate's vendored `kernel_mul_mv_id` was copied from (its
/// body, template shape, and `tgpig.z`-decode logic are byte-identical to
/// that commit's `ggml-metal.metal`), not assumed from the kernel body
/// alone or copied from `call_quantized_matmul_mv_t`'s own per-dtype table
/// (a *different*, non-`_id` kernel body that happens to share some but not
/// all tuning values -- see the `Q2K` case below).
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_id(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    src0_offset: usize,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    ids_shape: &[usize],
    ids_stride: &[usize],
    ids: &Buffer,
    ids_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Same shape convention as call_quantized_matmul_mm_id.
    let ne00 = src0_shape[src0_shape.len() - 1] as i64; // n_in (contraction dim)
    let ne01 = src0_shape[src0_shape.len() - 2] as i64; // n_out
    let ne02 = src0_shape[src0_shape.len() - 3] as i64; // n_expert
                                                        // Quantized src0's innermost stride is always 0 in ggml's own
                                                        // convention -- sub-block indexing happens inside the kernel from
                                                        // ne00/dtype, not from a passed byte stride. Matches
                                                        // call_quantized_matmul_mv_t's own `nb00 = 0i64` for the same reason.
    let nb00 = 0i64;
    let nb01 = src0_stride[src0_stride.len() - 2] as i64;
    let nb02 = src0_stride[src0_stride.len() - 3] as i64;

    let ne10 = src1_shape[src1_shape.len() - 1] as i64; // == ne00 (k), read from src1's own shape
    let nb10 = src1_stride[src1_stride.len() - 1] as i64;
    let ne11 = src1_shape[src1_shape.len() - 2] as i64; // n_expert_used (bcast) or 1
    let nb11 = src1_stride[src1_stride.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64; // n_tokens
    let nb12 = src1_stride[src1_stride.len() - 3] as i64;
    let ne13 = 1i64;

    let nei0 = ids_shape[ids_shape.len() - 1] as i64; // n_expert_used
    let nei1 = ids_shape[ids_shape.len() - 2] as i64; // n_tokens
    let nbi1 = ids_stride[ids_stride.len() - 2] as i64;

    let ne0 = dst_shape[dst_shape.len() - 1] as i64; // n_out
                                                     // Same load-bearing note as call_quantized_matmul_mm_id: this is the
                                                     // per-token row stride the kernel derives dst's write offset from
                                                     // (`dst + i1*ne0 + i2*ne1*ne0`), and must equal n_expert_used (nei0),
                                                     // not the total row count (nei0*nei1).
    let ne1 = nei0;
    let nb1 = ne0 * 4; // unused by the kernel; kept for ABI parity

    // Per-dtype (nth0, nth1, width_divisor, kernel name), reverse-derived
    // from ggml's `f3f65429`-era GGML_OP_MUL_MAT_ID mat-vec branch -- not
    // from call_quantized_matmul_mv_t's table, which diverges on Q2K (align
    // 4 there, for an unrelated plain-mv Metal bug fix) and F16/F32 (align 8
    // there vs. this kernel's own width=ne01 fallback). One match, not two
    // (tuning table + kernel name previously lived in separate match
    // statements over the same dtype, hand-synchronized on which variants
    // are unsupported -- re-enabling one of BF16/Q8_1/Q8K in only one of
    // the two would compile clean and panic the `unreachable!` in the
    // other the first time it was hit).
    let (nth0, nth1, divisor, name) = mv_id_dispatch_params(dtype)?;

    // Matches ggml's own host-side `GGML_ASSERT(ne00 >= nth0*nth1)` for
    // quantized src0 -- F16/F32 aren't gated by it upstream either.
    if !matches!(dtype, GgmlDType::F16 | GgmlDType::F32) && ne00 < (nth0 * nth1) as i64 {
        return Err(MetalKernelError::InvalidInput(format!(
            "qmm_mv_id: ne00 ({ne00}) must be >= nth0*nth1 ({})",
            nth0 * nth1
        )));
    }

    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, divisor),
        height: 1, // ggml's `_ne1` is always 1 for this op
        depth: (nei0 * nei1) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "qmm_mv_id {name} n_tokens={nei1} K={ne00} N={ne01} n_expert={ne02}"
    );

    // Buffer/scalar binding order follows the kernel's own declared
    // parameter list exactly (src0, src1, dst, ids, then every scalar in
    // declaration order) -- this is what candle's set_params! macro must
    // match, independent of how ggml's own host code happened to bind them
    // (one combined kargs struct vs. individual setBytes calls is an
    // Obj-C-side implementation detail of ggml's binding, not part of the
    // kernel's own ABI).
    set_params!(
        encoder,
        (
            (src0, src0_offset),
            (src1, src1_offset),
            Output::with_offset(dst, dst_offset),
            (ids, ids_offset),
            nei0,
            nei1,
            nbi1,
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
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

    // No threadgroup memory is set for any dtype this function supports --
    // confirmed against ggml's f3f65429 host code: only its IQ-family
    // kernels (none reachable here) call setThreadgroupMemoryLength for
    // this op. Do not copy call_quantized_matmul_mm_id's rowid-scan scratch
    // sizing formula; it belongs to a different kernel with a different
    // in-kernel scan this one doesn't do.
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Single source of truth for `call_quantized_matmul_mv_id`'s per-dtype
/// `(nth0, nth1, width_divisor, kernel_name)` -- shared by the dispatch
/// logic above and `mv_id_eligible` below so the two can't drift apart the
/// way two hand-synchronized `match dtype` statements over the same
/// variants otherwise could.
fn mv_id_dispatch_params(
    dtype: GgmlDType,
) -> Result<(usize, usize, usize, &'static str), MetalKernelError> {
    Ok(match dtype {
        GgmlDType::Q4_0 => (8, 8, 8, "kernel_mul_mv_id_q4_0_f32"),
        GgmlDType::Q4_1 => (8, 8, 8, "kernel_mul_mv_id_q4_1_f32"),
        GgmlDType::Q5_0 => (8, 8, 8, "kernel_mul_mv_id_q5_0_f32"),
        GgmlDType::Q5_1 => (8, 8, 8, "kernel_mul_mv_id_q5_1_f32"),
        GgmlDType::Q8_0 => (8, 8, 8, "kernel_mul_mv_id_q8_0_f32"),
        GgmlDType::Q2K => (2, 32, 8, "kernel_mul_mv_id_q2_K_f32"),
        GgmlDType::Q3K => (2, 32, 4, "kernel_mul_mv_id_q3_K_f32"),
        GgmlDType::Q4K => (4, 8, 4, "kernel_mul_mv_id_q4_K_f32"),
        GgmlDType::Q5K => (2, 32, 4, "kernel_mul_mv_id_q5_K_f32"),
        GgmlDType::Q6K => (2, 32, 2, "kernel_mul_mv_id_q6_K_f32"),
        GgmlDType::F16 => (32, 1, 1, "kernel_mul_mv_id_f16_f32"),
        GgmlDType::F32 => (32, 1, 1, "kernel_mul_mv_id_f32_f32"),
        GgmlDType::BF16 => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "BF16",
                "qmatmul_mv_id",
            ))
        }
        GgmlDType::Q8_1 => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "Q8_1",
                "qmatmul_mv_id",
            ))
        }
        GgmlDType::Q8K => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "Q8K",
                "qmatmul_mv_id",
            ))
        }
    })
}

/// Whether `call_quantized_matmul_mv_id` is production-eligible for
/// `dtype` at contraction-dimension `k` -- the single source of truth a
/// caller like `QMetalStorage::indexed_moe_forward` should use to decide
/// mv_id vs. mm_id, rather than duplicating a dtype allowlist or a k
/// threshold at the call site.
///
/// Narrower than "every dtype `call_quantized_matmul_mv_id` can technically
/// dispatch": Q4_1/Q5_0/Q5_1/Q8_0/Q3K/Q5K/F16/F32 all have real,
/// ground-truth-derived tuning entries in `mv_id_dispatch_params` (ggml
/// supports them, so the wrapper does too) but no differential test
/// coverage against `call_quantized_matmul_mm_id` yet -- they stay off
/// this list until they get one, even though calling the wrapper directly
/// with one of them would work correctly today.
///
/// Also enforces ggml's own `ne00 >= nth0*nth1` minimum contraction-dim
/// requirement (a real constraint of the mat-vec kernel, not an oversight):
/// below that threshold this returns `false` even for a covered dtype, so a
/// small-k caller falls back to `call_quantized_matmul_mm_id` (which has no
/// such minimum) instead of the wrapper's own hard `InvalidInput` error --
/// `call_quantized_matmul_mm_id` used to be called unconditionally for
/// every shape before this file existed, and callers relying on that
/// should see no new failure mode just because a `batch == 1` shape now
/// prefers mv_id.
pub fn mv_id_eligible(dtype: GgmlDType, k: usize) -> bool {
    let min_k = match dtype {
        GgmlDType::Q4K => 4 * 8,
        GgmlDType::Q6K => 2 * 32,
        GgmlDType::Q4_0 => 8 * 8,
        GgmlDType::Q2K => 2 * 32,
        _ => return false,
    };
    k >= min_k
}

fn divide(m: usize, b: usize) -> usize {
    m.div_ceil(b)
}
