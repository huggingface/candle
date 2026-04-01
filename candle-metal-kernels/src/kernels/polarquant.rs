use crate::utils::EncoderProvider;
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

/// Dispatch the PolarQuant fused centroid-dot kernel on Metal.
///
/// Computes `output[b][j] = norms[j] * Σ_i x_rot[b][i] * centroids[unpack(indices, j*d+i)]`
/// entirely on GPU. No CPU↔GPU transfers needed during inference.
///
/// # Arguments
/// * `x_rot` — pre-rotated input buffer, `[batch, d]` f32
/// * `indices` — bit-packed centroid indices, `[n_out * d / indices_per_byte]` u8
/// * `norms` — per-output-row weight norms, `[n_out]` f32
/// * `centroids` — Lloyd-Max centroid table, `[2^bit_width]` f32
/// * `output` — output buffer, `[batch, n_out]` f32
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
        1 => "polarquant_centroid_dot_1bit",
        2 => "polarquant_centroid_dot_2bit",
        3 => "polarquant_centroid_dot_3bit",
        4 => "polarquant_centroid_dot_4bit",
        5 => "polarquant_centroid_dot_5bit",
        6 => "polarquant_centroid_dot_6bit",
        7 => "polarquant_centroid_dot_7bit",
        _ => {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "PolarQuant: unsupported bit_width {bit_width}"
            )))
        }
    };

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

    let thread_group_size = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    let grid_size = MTLSize {
        width: n_out,
        height: batch,
        depth: 1,
    };

    encoder.dispatch_threads(grid_size, thread_group_size);

    Ok(())
}
