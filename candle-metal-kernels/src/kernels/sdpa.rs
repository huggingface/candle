use crate::utils::EncoderProvider;
use crate::{
    set_params, Buffer, ComputeCommandEncoder, ConstantValues, Device, EncoderParam, Kernels,
    MetalKernelError, Source, Value,
};
use objc2_metal::{MTLResourceUsage, MTLSize};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum SdpaDType {
    BF16,
    F16,
    F32,
}

/// SDPA full is supported when:
/// - q head dim == 64, 128
/// - no mask
/// - q heads == kv heads
/// - final type != bf16 (TODO maybe just template this kernel too?)
/// - q,k,v are contiguous
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_full(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_buffer: &Buffer,
    v_offset: usize,
    v_buffer: &Buffer,
    output: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct MLXFastAttentionParams {
        m: i32,
        n: i32,
        k: i32,

        ldq: i32, // ldq == ldo
        ldk: i32,
        ldv: i32,
        lds: i32,
        ldo: i32,

        tiles_n: i32,
        tiles_m: i32,

        batch_stride_q: i32,
        batch_stride_k: i32,
        batch_stride_v: i32,
        batch_stride_o: i32,

        swizzle_log: i32,
        gemm_n_iterations_aligned: i32,
        gemm_k_iterations_aligned: i32,
        gemm_sv_m_block_iterations: i32,

        batch_ndim: i32,
        alpha: f32,
        softcapping: f32,
    }

    let bk = q_shape.last().unwrap();

    const BN: usize = 16;
    const BM: usize = 16;
    const WM: usize = 2;
    const WN: usize = 2;

    let name = match (bk, itype) {
        (32, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_32_itype_half",
        (64, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_64_itype_half",
        (96, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_96_itype_half",
        (128, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_128_itype_half",
        (256, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_256_itype_half",
        (32, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_32_itype_float",
        (64, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_64_itype_float",
        (96, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_96_itype_float",
        (128, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_128_itype_float",
        (256, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_256_itype_float",
        (other, SdpaDType::F16 | SdpaDType::F32) => {
            return Err(MetalKernelError::SdpaHeadSizeMismatch {
                variation: "full",
                got: *other,
                expected: vec![32, 64, 96, 128, 256],
            })
        }
        (_, SdpaDType::BF16) => {
            return Err(MetalKernelError::SdpaHeadDTypeMismatch {
                variation: "full",
                got: SdpaDType::BF16,
            })
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::Sdpa, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // q = (bs, qhead, seq, hidden)
    // k/v = (bs, kv_head, seq, hidden)

    let qseq = q_shape[q_shape.len() - 2];

    let m = q_shape[q_shape.len() - 2];
    let n = m;
    let k = q_shape[q_shape.len() - 1];
    let bs_out = q_shape[0] * q_shape[1];

    let batch_shape = [q_shape[0] * q_shape[1]];
    let dk = q_shape[q_shape.len() - 1];
    let ldq = dk;
    let ldk = dk;
    let ldv = dk;
    let lds = BN;
    let ldo = dk;

    let tn = 1;
    let tm = m.div_ceil(BM);

    let b_stride_q = dk * qseq;
    let b_stride_k = dk * qseq;
    let b_stride_v = dk * qseq;
    let b_stride_o = dk * qseq;
    let swizzle_log = 0;
    let gemm_n_iterations_aligned = n.div_ceil(BN);
    let gemm_k_iterations_aligned = k.div_ceil(*bk);
    let gemm_sv_m_block_iterations = m.div_ceil(BM);
    let batch_ndim = batch_shape.len();

    let alpha = if softcapping != 1. {
        alpha / softcapping
    } else {
        alpha
    };

    let params = MLXFastAttentionParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        ldq: ldq as i32,
        ldk: ldk as i32,
        ldv: ldv as i32,
        lds: lds as i32,
        ldo: ldo as i32,
        tiles_n: tn,
        tiles_m: tm as i32,
        batch_stride_q: b_stride_q as i32,
        batch_stride_k: b_stride_k as i32,
        batch_stride_v: b_stride_v as i32,
        batch_stride_o: b_stride_o as i32,
        swizzle_log,
        gemm_n_iterations_aligned: gemm_n_iterations_aligned as i32,
        gemm_k_iterations_aligned: gemm_k_iterations_aligned as i32,
        gemm_sv_m_block_iterations: gemm_sv_m_block_iterations as i32,
        batch_ndim: batch_ndim as i32,
        alpha,
        softcapping,
    };
    let batch_strides = [b_stride_q, b_stride_k, b_stride_v, b_stride_o];

    impl EncoderParam for MLXFastAttentionParams {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            output,
            params,
            &batch_shape[..],
            &batch_strides[..]
        )
    );

    let grid_dims = MTLSize {
        width: 1,
        height: tm,
        depth: bs_out,
    };
    let group_dims = MTLSize {
        width: 32,
        height: WM,
        depth: WN,
    };
    encoder.use_resource(q_buffer, MTLResourceUsage::Read);
    encoder.use_resource(k_buffer, MTLResourceUsage::Read);
    encoder.use_resource(v_buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// SDPA full is supported when:
/// - q head dim == 64, 96, 128
/// - no mask
/// - q,k,v are contiguous
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    output: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    let bk = q_shape.last().unwrap();

    let gqa_factor = (q_shape[1] / k_shape[1]) as i32;
    let n = k_shape[2] as i32;
    let b = (q_shape[0] * q_shape[1]) as i32;
    let kstride = k_stride[1];
    let vstride = v_stride[1];

    let name = match (bk, itype) {
        (32, SdpaDType::F16) => "sdpa_vector_float16_t_32",
        (64, SdpaDType::F16) => "sdpa_vector_float16_t_64",
        (96, SdpaDType::F16) => "sdpa_vector_float16_t_96",
        (128, SdpaDType::F16) => "sdpa_vector_float16_t_128",
        (256, SdpaDType::F16) => "sdpa_vector_float16_t_256",
        (32, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_32",
        (64, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_64",
        (96, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_96",
        (128, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_128",
        (256, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_256",
        (32, SdpaDType::F32) => "sdpa_vector_float_32",
        (64, SdpaDType::F32) => "sdpa_vector_float_64",
        (96, SdpaDType::F32) => "sdpa_vector_float_96",
        (128, SdpaDType::F32) => "sdpa_vector_float_128",
        (256, SdpaDType::F32) => "sdpa_vector_float_256",
        (other, _) => {
            return Err(MetalKernelError::SdpaHeadSizeMismatch {
                variation: "vector",
                got: *other,
                expected: vec![32, 64, 96, 128, 256],
            })
        }
    };

    let alpha = if softcapping != 1. {
        alpha / softcapping
    } else {
        alpha
    };

    let constants = Some(ConstantValues::new(vec![(
        20,
        Value::Bool(/* sdpa_vector_has_mask */ false),
    )]));

    let pipeline = kernels.load_pipeline_with_constants(device, Source::Sdpa, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // q = (bs, qhead, seq, hidden)
    // k/v = (bs, kv_head, kv_seq, hidden)

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            output,
            gqa_factor,
            n,
            kstride,
            vstride,
            alpha,
            softcapping
        )
    );

    let grid_dims = MTLSize {
        width: 1,
        height: b as usize,
        depth: 1,
    };
    let group_dims = MTLSize {
        width: 1024,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(q_buffer, MTLResourceUsage::Read);
    encoder.use_resource(k_buffer, MTLResourceUsage::Read);
    encoder.use_resource(v_buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

pub const SDPA_2PASS_BLOCKS: usize = 32;

/// SDPA vector 2pass is supported when:
/// - q head dim == 64, 96, 128
/// - no mask
/// - q,k,v are contiguous
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_2pass(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    output: &Buffer,
    intermediate: &Buffer,
    sums: &Buffer,
    maxs: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    let bk = q_shape.last().unwrap();

    // First pass
    {
        let name_pass1 = match (bk, itype) {
            (32, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_32",
            (64, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_64",
            (96, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_96",
            (128, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_128",
            (256, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_256",
            (32, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_32",
            (64, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_64",
            (96, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_96",
            (128, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_128",
            (256, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_256",
            (32, SdpaDType::F32) => "sdpa_vector_2pass_1_float_32",
            (64, SdpaDType::F32) => "sdpa_vector_2pass_1_float_64",
            (96, SdpaDType::F32) => "sdpa_vector_2pass_1_float_96",
            (128, SdpaDType::F32) => "sdpa_vector_2pass_1_float_128",
            (256, SdpaDType::F32) => "sdpa_vector_2pass_1_float_256",
            (other, _) => {
                return Err(MetalKernelError::SdpaHeadSizeMismatch {
                    variation: "vector_2pass_1",
                    got: *other,
                    expected: vec![32, 64, 96, 128, 256],
                })
            }
        };

        let gqa_factor = (q_shape[1] / k_shape[1]) as i32;
        let n = k_shape[2] as i32;
        let b = (q_shape[0] * q_shape[1]) as i32;
        let kstride = k_stride[1];
        let vstride = v_stride[1];

        let alpha = if softcapping != 1. {
            alpha / softcapping
        } else {
            alpha
        };

        let constants = Some(ConstantValues::new(vec![(
            20,
            Value::Bool(/* sdpa_vector_has_mask */ false),
        )]));

        let pipeline =
            kernels.load_pipeline_with_constants(device, Source::Sdpa, name_pass1, constants)?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        // q = (bs, qhead, seq, hidden)
        // k/v = (bs, kv_head, kv_seq, hidden)

        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                intermediate,
                sums,
                maxs,
                gqa_factor,
                n,
                kstride,
                vstride,
                alpha,
                softcapping
            )
        );

        let grid_dims = MTLSize {
            width: 1,
            height: b as usize,
            depth: SDPA_2PASS_BLOCKS,
        };
        let group_dims = MTLSize {
            width: 8 * 32,
            height: 1,
            depth: 1,
        };
        encoder.use_resource(q_buffer, MTLResourceUsage::Read);
        encoder.use_resource(k_buffer, MTLResourceUsage::Read);
        encoder.use_resource(v_buffer, MTLResourceUsage::Read);
        encoder.use_resource(intermediate, MTLResourceUsage::Write);
        encoder.use_resource(sums, MTLResourceUsage::Write);
        encoder.use_resource(maxs, MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }

    // Final pass
    {
        let name_pass2 = match (bk, itype) {
            (32, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_32",
            (64, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_64",
            (96, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_96",
            (128, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_128",
            (256, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_256",
            (32, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_32",
            (64, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_64",
            (96, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_96",
            (128, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_128",
            (256, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_256",
            (32, SdpaDType::F32) => "sdpa_vector_2pass_2_float_32",
            (64, SdpaDType::F32) => "sdpa_vector_2pass_2_float_64",
            (96, SdpaDType::F32) => "sdpa_vector_2pass_2_float_96",
            (128, SdpaDType::F32) => "sdpa_vector_2pass_2_float_128",
            (256, SdpaDType::F32) => "sdpa_vector_2pass_2_float_256",
            (other, _) => {
                return Err(MetalKernelError::SdpaHeadSizeMismatch {
                    variation: "vector_2pass_2",
                    got: *other,
                    expected: vec![32, 64, 96, 128, 256],
                })
            }
        };

        let b = q_shape[0] * q_shape[1];

        let pipeline = kernels.load_pipeline(device, Source::Sdpa, name_pass2)?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        // q = (bs, qhead, seq, hidden)
        // k/v = (bs, kv_head, kv_seq, hidden)

        set_params!(encoder, (intermediate, sums, maxs, output));

        let grid_dims = MTLSize {
            width: 1,
            height: b,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        encoder.use_resource(intermediate, MTLResourceUsage::Write);
        encoder.use_resource(sums, MTLResourceUsage::Write);
        encoder.use_resource(maxs, MTLResourceUsage::Write);
        encoder.use_resource(output, MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }
    Ok(())
}
