use crate::utils::EncoderProvider;
use crate::{ConstantValues, Kernels, MetalKernelError, Source, Value};
use metal::{Buffer, ComputeCommandEncoderRef, Device, MTLSize, NSUInteger};
use std::ffi::c_void;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum GemmDType {
    BF16,
    F16,
    F32,
}

#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct GemmParams {
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldd: i32,
        tiles_n: i32,
        tiles_m: i32,
        batch_stride_a: isize,
        batch_stride_b: isize,
        batch_stride_d: isize,
        swizzle_log: i32,
        gemm_k_iterations_aligned: i32,
        batch_ndim: i32,
    }
    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // lhs has shape b, m, k
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        })?;
    };
    // rhs has shape b, k, n
    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        })?;
    };
    let (bm, bn, bk, wn, wm) = (32, 32, 16, 2, 2);
    // https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/matmul.cpp#L422
    let constants = Some(ConstantValues::new(vec![
        (10, Value::Bool(/* has_batch */ b > 1)),
        (100, Value::Bool(/* use_out_source */ false)),
        (110, Value::Bool(/* do_axpby */ false)),
        (200, Value::Bool(/* align_m */ m % bm == 0)),
        (201, Value::Bool(/* align_n */ n % bn == 0)),
        (202, Value::Bool(/* align_k */ k % bk == 0)),
        (300, Value::Bool(/* do_gather */ false)),
    ]));

    let swizzle_log = 0;
    let tile = 1 << swizzle_log;
    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let tn = tn * tile;
    let tm = tm.div_ceil(tile);

    let batch_stride_a = if lhs_stride.len() > 2 {
        lhs_stride[lhs_stride.len() - 3]
    } else {
        m * k
    };
    let batch_stride_b = if rhs_stride.len() > 2 {
        rhs_stride[rhs_stride.len() - 3]
    } else {
        n * k
    };

    let gemm_params = GemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda,
        ldb,
        ldd: n as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        swizzle_log,
        batch_stride_a: batch_stride_a as isize,
        batch_stride_b: batch_stride_b as isize,
        batch_stride_d: (m * n) as isize,
        batch_ndim: 1i32,
        gemm_k_iterations_aligned: (k / bk) as i32,
    };
    let batch_strides = [gemm_params.batch_stride_a, gemm_params.batch_stride_b];

    // TODO(laurent): generate the name
    // template [[host_name("gemm_" #tname "_"  #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn)]]
    let name = match (dtype, a_trans, b_trans) {
        (GemmDType::F32, false, false) => "gemm_nn_f32_f32_32_32_16_2_2",
        (GemmDType::F32, true, false) => "gemm_tn_f32_f32_32_32_16_2_2",
        (GemmDType::F32, false, true) => "gemm_nt_f32_f32_32_32_16_2_2",
        (GemmDType::F32, true, true) => "gemm_tt_f32_f32_32_32_16_2_2",
        (GemmDType::BF16, false, false) => "gemm_nn_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, true, false) => "gemm_tn_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, false, true) => "gemm_nt_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, true, true) => "gemm_tt_bf16_bf16_32_32_16_2_2",
        (GemmDType::F16, false, false) => "gemm_nn_f16_f16_32_32_16_2_2",
        (GemmDType::F16, true, false) => "gemm_tn_f16_f16_32_32_16_2_2",
        (GemmDType::F16, false, true) => "gemm_nt_f16_f16_32_32_16_2_2",
        (GemmDType::F16, true, true) => "gemm_tt_f16_f16_32_32_16_2_2",
    };
    let pipeline = kernels.load_pipeline_with_constants(device, Source::Gemm, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as NSUInteger);
    encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as NSUInteger);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_bytes(
        4,
        std::mem::size_of::<GemmParams>() as u64,
        &gemm_params as *const GemmParams as *const c_void,
    );
    encoder.set_bytes(
        6, // batch_shape
        std::mem::size_of::<i32>() as u64,
        &(b as i32) as *const i32 as *const c_void,
    );
    encoder.set_bytes(
        7,
        (std::mem::size_of::<isize>() * batch_strides.len()) as u64,
        batch_strides.as_ptr() as *const c_void,
    );

    let grid_size = MTLSize {
        width: tn as u64,
        height: tm as u64,
        depth: /* batch_size_out */ b as u64,
    };
    let group_size = MTLSize {
        width: 32,
        height: wn,
        depth: wm,
    };
    encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);
    Ok(())
}
