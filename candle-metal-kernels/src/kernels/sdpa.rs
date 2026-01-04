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
    q_strides: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_strides: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_buffer: &Buffer,
    v_strides: &[usize],
    mask_type: Option<SdpaDType>,
    mask_buffer: Option<&Buffer>,
    m_strides: Option<&[usize]>,
    output: &Buffer,
    o_strides: &[usize],
    scale: f32,
    softcapping: f32,
    do_causal: bool,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct AttnParams {
        b: i32,
        h: i32,
        d: i32,
        ql: i32,
        kl: i32,
        gqa_factor: i32,
        scale: f32,
        softcapping: f32, // Must match Metal struct layout (1.0 = disabled)
        nq: i32,
        nk: i32,
        nq_aligned: i32,
        nk_aligned: i32,
        ql_rem: i32,
        kl_rem: i32,
        ql_off: i32,
        q_strides: [i64; 3],
        k_strides: [i64; 3],
        v_strides: [i64; 3],
        o_strides: [i64; 3],
    }

    #[derive(Debug)]
    #[repr(C)]
    struct AttnMaskParams {
        m_strides: [i64; 3],
    }

    const WN: usize = 1;

    let bd = q_shape[q_shape.len() - 1];
    if ![16, 32, 64, 72, 80, 96, 128, 256].contains(&bd) {
        return Err(MetalKernelError::SdpaHeadSizeMismatch {
            variation: "full",
            got: bd,
            expected: vec![16, 32, 64, 72, 80, 96, 128, 256],
        });
    };

    // Updated blocking logic
    // For small head dims (32, 64), we use larger query blocks (64) to improve bandwidth
    let (bq, bk) = if bd <= 64 {
        (64, 32)
    } else if bd < 128 {
        (32, 32)
    } else {
        (32, 16)
    };

    // TQ = bq / (WM * WN * kFragSize) must be 1 => WM = bq / (WN * kFragSize) = bq / 8
    let wm = match bq {
        64 => 8,
        _ => 4,
    };

    let b = q_shape[0];
    let h = q_shape[1];
    let d = q_shape[3];
    let gqa_factor = q_shape[1] / k_shape[1];

    let ql = q_shape[2];
    let kl = k_shape[2];

    let align_q = (ql % bq) == 0;
    let align_k = (kl % bk) == 0;
    let has_mask = mask_buffer.is_some();

    let itype_repr = match itype {
        SdpaDType::BF16 => "bfloat16",
        SdpaDType::F16 => "float16",
        SdpaDType::F32 => "float32",
    };
    let mask_repr = match mask_type {
        Some(SdpaDType::BF16) => "bfloat16",
        Some(SdpaDType::F16) => "float16",
        Some(SdpaDType::F32) => "float32",
        None => itype_repr,
    };
    let name =
        format!("steel_attention_{itype_repr}_bq{bq}_bk{bk}_bd{bd}_wm{wm}_wn{WN}_mask{mask_repr}");

    let constants = Some(ConstantValues::new(vec![
        (200, Value::Bool(/* align_Q */ align_q)),
        (201, Value::Bool(/* align_K */ align_k)),
        (300, Value::Bool(/* has_mask */ has_mask)),
        (301, Value::Bool(/* do_causal */ do_causal)),
    ]));

    let pipeline = kernels.load_pipeline_with_constants(device, Source::Sdpa, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let nq = (ql + bq - 1) / bq;
    let nk = (kl + bk - 1) / bk;

    let nq_aligned = ql / bq;
    let nk_aligned = kl / bk;

    let params = AttnParams {
        b: b as i32,
        h: h as i32,
        d: d as i32,
        ql: ql as i32,
        kl: kl as i32,
        gqa_factor: gqa_factor as i32,
        scale,
        softcapping: 1.0, // SDPA full doesn't support softcapping, always 1.0
        nq: nq as i32,
        nk: nk as i32,
        nq_aligned: nq_aligned as i32,
        nk_aligned: nk_aligned as i32,
        ql_rem: ql.wrapping_sub(nq_aligned * bq) as i32,
        kl_rem: kl.wrapping_sub(nk_aligned * bk) as i32,
        ql_off: kl.wrapping_sub(ql) as i32,
        q_strides: [
            q_strides[0] as i64,
            q_strides[1] as i64,
            q_strides[2] as i64,
        ],
        k_strides: [
            k_strides[0] as i64,
            k_strides[1] as i64,
            k_strides[2] as i64,
        ],
        v_strides: [
            v_strides[0] as i64,
            v_strides[1] as i64,
            v_strides[2] as i64,
        ],
        o_strides: [
            o_strides[0] as i64,
            o_strides[1] as i64,
            o_strides[2] as i64,
        ],
    };

    impl EncoderParam for AttnParams {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    impl EncoderParam for AttnMaskParams {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    if let Some(mask) = mask_buffer {
        let mask_strides = m_strides.unwrap();
        let mask_params = AttnMaskParams {
            m_strides: [
                mask_strides[0] as i64,
                mask_strides[1] as i64,
                mask_strides[2] as i64,
            ],
        };
        encoder.use_resource(mask, MTLResourceUsage::Read);

        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                output,
                params,
                mask_params,
                mask
            )
        );
    } else {
        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                output,
                params
            )
        );
    }

    let grid_dims = MTLSize {
        width: nq,
        height: h,
        depth: b,
    };
    let group_dims = MTLSize {
        width: 32,
        height: wm,
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

    let (name, grid_height) = match (bk, itype) {
        (16, SdpaDType::F16) => ("sdpa_vector_p16_float16_t", (b as usize + 1) / 2),
        (32, SdpaDType::F16) => ("sdpa_vector_float16_t_32", b as usize),
        (64, SdpaDType::F16) => ("sdpa_vector_float16_t_64", b as usize),
        (96, SdpaDType::F16) => ("sdpa_vector_float16_t_96", b as usize),
        (128, SdpaDType::F16) => ("sdpa_vector_float16_t_128", b as usize),
        (256, SdpaDType::F16) => ("sdpa_vector_float16_t_256", b as usize),
        (16, SdpaDType::BF16) => ("sdpa_vector_p16_bfloat16_t", (b as usize + 1) / 2),
        (32, SdpaDType::BF16) => ("sdpa_vector_bfloat16_t_32", b as usize),
        (64, SdpaDType::BF16) => ("sdpa_vector_bfloat16_t_64", b as usize),
        (96, SdpaDType::BF16) => ("sdpa_vector_bfloat16_t_96", b as usize),
        (128, SdpaDType::BF16) => ("sdpa_vector_bfloat16_t_128", b as usize),
        (256, SdpaDType::BF16) => ("sdpa_vector_bfloat16_t_256", b as usize),
        (16, SdpaDType::F32) => ("sdpa_vector_p16_float", (b as usize + 1) / 2),
        (32, SdpaDType::F32) => ("sdpa_vector_float_32", b as usize),
        (64, SdpaDType::F32) => ("sdpa_vector_float_64", b as usize),
        (96, SdpaDType::F32) => ("sdpa_vector_float_96", b as usize),
        (128, SdpaDType::F32) => ("sdpa_vector_float_128", b as usize),
        (256, SdpaDType::F32) => ("sdpa_vector_float_256", b as usize),
        (other, _) => {
            return Err(MetalKernelError::SdpaHeadSizeMismatch {
                variation: "vector",
                got: *other,
                expected: vec![16, 32, 64, 96, 128, 256],
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
        height: grid_height,
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
            (16, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_16",
            (32, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_32",
            (64, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_64",
            (96, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_96",
            (128, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_128",
            (256, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_256",
            (16, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_16",
            (32, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_32",
            (64, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_64",
            (96, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_96",
            (128, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_128",
            (256, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_256",
            (16, SdpaDType::F32) => "sdpa_vector_2pass_1_float_16",
            (32, SdpaDType::F32) => "sdpa_vector_2pass_1_float_32",
            (64, SdpaDType::F32) => "sdpa_vector_2pass_1_float_64",
            (96, SdpaDType::F32) => "sdpa_vector_2pass_1_float_96",
            (128, SdpaDType::F32) => "sdpa_vector_2pass_1_float_128",
            (256, SdpaDType::F32) => "sdpa_vector_2pass_1_float_256",
            (other, _) => {
                return Err(MetalKernelError::SdpaHeadSizeMismatch {
                    variation: "vector_2pass_1",
                    got: *other,
                    expected: vec![16, 32, 64, 96, 128, 256],
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
            (16, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_16",
            (32, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_32",
            (64, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_64",
            (96, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_96",
            (128, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_128",
            (256, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_256",
            (16, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_16",
            (32, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_32",
            (64, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_64",
            (96, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_96",
            (128, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_128",
            (256, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_256",
            (16, SdpaDType::F32) => "sdpa_vector_2pass_2_float_16",
            (32, SdpaDType::F32) => "sdpa_vector_2pass_2_float_32",
            (64, SdpaDType::F32) => "sdpa_vector_2pass_2_float_64",
            (96, SdpaDType::F32) => "sdpa_vector_2pass_2_float_96",
            (128, SdpaDType::F32) => "sdpa_vector_2pass_2_float_128",
            (256, SdpaDType::F32) => "sdpa_vector_2pass_2_float_256",
            (other, _) => {
                return Err(MetalKernelError::SdpaHeadSizeMismatch {
                    variation: "vector_2pass_2",
                    got: *other,
                    expected: vec![16, 32, 64, 96, 128, 256],
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

#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_full_with_rel_pos(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_strides: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_strides: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_buffer: &Buffer,
    v_strides: &[usize],
    output: &Buffer,
    o_strides: &[usize],
    term_h: &Buffer,
    term_w: &Buffer,
    h_dim: i32,
    w_dim: i32,
    scale: f32,
    // softcapping: f32, // Not supported in full SDPA yet
    // do_causal: bool, // Not supported in this fused kernel yet (assumed false for SAM3 backbone)
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct AttnParams {
        b: i32,
        h: i32,
        d: i32,
        ql: i32,
        kl: i32,
        gqa_factor: i32,
        scale: f32,
        softcapping: f32,
        nq: i32,
        nk: i32,
        nq_aligned: i32,
        nk_aligned: i32,
        ql_rem: i32,
        kl_rem: i32,
        ql_off: i32,
        q_strides: [i64; 3],
        k_strides: [i64; 3],
        v_strides: [i64; 3],
        o_strides: [i64; 3],
    }

    const WN: usize = 1;

    let bd = q_shape[q_shape.len() - 1];
    if ![16, 32, 64, 72, 80, 96, 128, 256].contains(&bd) {
        return Err(MetalKernelError::SdpaHeadSizeMismatch {
            variation: "full_rel_pos",
            got: bd,
            expected: vec![16, 32, 64, 72, 80, 96, 128, 256],
        });
    };

    // Blocking logic (same as full)
    let (bq, bk) = if bd <= 64 {
        (64, 32)
    } else if bd < 128 {
        (32, 32)
    } else {
        (32, 16)
    };

    let wm = match bq {
        64 => 8,
        _ => 4,
    };

    let b = q_shape[0];
    let h = q_shape[1];
    let d = q_shape[3];
    let gqa_factor = q_shape[1] / k_shape[1];

    let ql = q_shape[2];
    let kl = k_shape[2];

    let align_q = (ql % bq) == 0;
    let align_k = (kl % bk) == 0;

    // Check mask shape/validity? Handled by caller.

    let itype_repr = match itype {
        SdpaDType::BF16 => "bfloat16",
        SdpaDType::F16 => "float16",
        SdpaDType::F32 => "float32",
    };

    let name = format!("steel_attention_rel_pos_{itype_repr}_bq{bq}_bk{bk}_bd{bd}_wm{wm}_wn{WN}");

    let constants = Some(ConstantValues::new(vec![
        (200, Value::Bool(/* align_Q */ align_q)),
        (201, Value::Bool(/* align_K */ align_k)),
        (300, Value::Bool(/* has_mask */ false)), // We don't use the generic mask path
        (301, Value::Bool(/* do_causal */ false)),
    ]));

    let pipeline = kernels.load_pipeline_with_constants(device, Source::Sdpa, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let nq = (ql + bq - 1) / bq;
    let nk = (kl + bk - 1) / bk;

    let nq_aligned = ql / bq;
    let nk_aligned = kl / bk;

    let params = AttnParams {
        b: b as i32,
        h: h as i32,
        d: d as i32,
        ql: ql as i32,
        kl: kl as i32,
        gqa_factor: gqa_factor as i32,
        scale,
        softcapping: 1.0,
        nq: nq as i32,
        nk: nk as i32,
        nq_aligned: nq_aligned as i32,
        nk_aligned: nk_aligned as i32,
        ql_rem: ql.wrapping_sub(nq_aligned * bq) as i32,
        kl_rem: kl.wrapping_sub(nk_aligned * bk) as i32,
        ql_off: kl.wrapping_sub(ql) as i32,
        q_strides: [
            q_strides[0] as i64,
            q_strides[1] as i64,
            q_strides[2] as i64,
        ],
        k_strides: [
            k_strides[0] as i64,
            k_strides[1] as i64,
            k_strides[2] as i64,
        ],
        v_strides: [
            v_strides[0] as i64,
            v_strides[1] as i64,
            v_strides[2] as i64,
        ],
        o_strides: [
            o_strides[0] as i64,
            o_strides[1] as i64,
            o_strides[2] as i64,
        ],
    };

    impl EncoderParam for AttnParams {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    encoder.use_resource(term_h, MTLResourceUsage::Read);
    encoder.use_resource(term_w, MTLResourceUsage::Read);

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            output,
            params,
            term_h,
            term_w,
            h_dim,
            w_dim
        )
    );

    let grid_dims = MTLSize {
        width: nq,
        height: h,
        depth: b,
    };
    let group_dims = MTLSize {
        width: 32,
        height: wm,
        depth: WN,
    };
    encoder.use_resource(q_buffer, MTLResourceUsage::Read);
    encoder.use_resource(k_buffer, MTLResourceUsage::Read);
    encoder.use_resource(v_buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_dims, group_dims);

    Ok(())
}
