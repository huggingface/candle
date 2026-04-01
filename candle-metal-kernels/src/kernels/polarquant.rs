use crate::utils::EncoderProvider;
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

fn divide(m: usize, b: usize) -> usize {
    (m + b - 1) / b
}

/// Dispatch the PolarQuant SIMD-optimized centroid-dot kernel on Metal.
///
/// Uses SIMD groups (32 threads) with 4 output rows per group,
/// matching GGML's mul_vec_q_n pattern for bandwidth-optimal access.
#[allow(clippy::too_many_arguments)]
pub fn call_polarquant_centroid_dot(
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
                "PolarQuant: unsupported bit_width {bit_width}"
            )))
        }
    };

    const N_DST: usize = 4;
    const N_SIMDGROUP: usize = 2;
    const SIMD_WIDTH: usize = 32;

    let pipeline = kernels.load_pipeline(device, Source::PolarQuant, name)?;
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
