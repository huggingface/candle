use crate::utils::EncoderProvider;
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

fn divide(m: usize, b: usize) -> usize {
    (m + b - 1) / b
}

/// Dispatch the TurboQuantMse SIMD-optimized centroid-dot kernel on Metal.
///
/// Uses SIMD groups (32 threads) with 4 output rows per group,
/// matching GGML's mul_vec_q_n pattern for bandwidth-optimal access.
#[allow(clippy::too_many_arguments)]
pub fn call_turboquant_mse_centroid_dot(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    bit_width: usize,
    d: usize,
    n_out: usize,
    batch: usize,
    x_rot: &Buffer,
    x_rot_offset: usize,
    indices: &Buffer,
    norms: &Buffer,
    centroids: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match bit_width {
        1 => "polarquant_mv_f32_1bit",
        2 => "polarquant_mv_f32_2bit",
        3 => "polarquant_mv_f32_3bit",
        4 => "polarquant_mv_f32_4bit",
        5 => "polarquant_mv_f32_5bit",
        6 => "polarquant_mv_f32_6bit",
        7 => "polarquant_mv_f32_7bit",
        _ => {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "TurboQuantMse: unsupported bit_width {bit_width}"
            )))
        }
    };

    const N_DST: usize = 8;
    const N_SIMDGROUP: usize = 2;
    const SIMD_WIDTH: usize = 32;

    let pipeline = kernels.load_pipeline(device, Source::TurboQuantMse, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let d_u32 = d as u32;
    let n_out_u32 = n_out as u32;
    let batch_u32 = batch as u32;

    set_params!(
        encoder,
        (
            (x_rot, x_rot_offset),
            indices,
            norms,
            centroids,
            output,
            d_u32,
            n_out_u32,
            batch_u32
        )
    );

    encoder.use_resource(x_rot, MTLResourceUsage::Read);
    encoder.use_resource(indices, MTLResourceUsage::Read);
    encoder.use_resource(norms, MTLResourceUsage::Read);
    encoder.use_resource(centroids, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let n_row_groups = divide(n_out, N_DST * N_SIMDGROUP);
    let thread_groups_count = MTLSize {
        width: n_row_groups,
        height: batch,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: SIMD_WIDTH,
        height: N_SIMDGROUP,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);

    Ok(())
}

/// Dispatch the fused Hadamard + centroid-dot kernel.
/// No separate rotation matmul needed — the Hadamard transform happens
/// in threadgroup shared memory before the centroid dot product.
#[allow(clippy::too_many_arguments)]
pub fn call_turboquant_mse_fused_hadamard(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    bit_width: usize,
    d: usize,
    n_out: usize,
    batch: usize,
    x: &Buffer,
    x_offset: usize,
    indices: &Buffer,
    norms: &Buffer,
    centroids: &Buffer,
    signs: &Buffer,
    scale: f32,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match bit_width {
        1 => "polarquant_mv_fused_hadamard_1bit",
        2 => "polarquant_mv_fused_hadamard_2bit",
        3 => "polarquant_mv_fused_hadamard_3bit",
        4 => "polarquant_mv_fused_hadamard_4bit",
        5 => "polarquant_mv_fused_hadamard_5bit",
        6 => "polarquant_mv_fused_hadamard_6bit",
        7 => "polarquant_mv_fused_hadamard_7bit",
        _ => {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "TurboQuantMse fused Hadamard: unsupported bit_width {bit_width}"
            )))
        }
    };

    const N_DST: usize = 8;
    const THREADS_PER_TG: usize = 256;
    let n_sg = THREADS_PER_TG / 32;

    let pipeline = kernels.load_pipeline(device, Source::TurboQuantMse, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let d_u32 = d as u32;
    let n_out_u32 = n_out as u32;
    let batch_u32 = batch as u32;

    set_params!(
        encoder,
        (
            (x, x_offset),
            indices,
            norms,
            centroids,
            output,
            signs,
            d_u32,
            n_out_u32,
            batch_u32,
            scale
        )
    );

    encoder.use_resource(x, MTLResourceUsage::Read);
    encoder.use_resource(indices, MTLResourceUsage::Read);
    encoder.use_resource(norms, MTLResourceUsage::Read);
    encoder.use_resource(centroids, MTLResourceUsage::Read);
    encoder.use_resource(signs, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    // Threadgroup shared memory for the input (d floats)
    encoder.set_threadgroup_memory_length(0, d * 4);

    let n_row_groups = divide(n_out, N_DST * n_sg);
    let thread_groups_count = MTLSize {
        width: n_row_groups,
        height: batch,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: THREADS_PER_TG,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);

    Ok(())
}
