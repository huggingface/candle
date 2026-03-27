//! Deformable Convolution 2D Metal kernel bindings

use crate::utils::{BufferOffset, EncoderProvider};
use crate::{linear_split, set_params, Buffer, Device, Kernels, MetalKernelError, Source};
use objc2_metal::MTLResourceUsage;

/// Call deformable im2col kernel
///
/// This kernel only performs the im2col step with deformable sampling.
/// The GEMM step should be done using Tensor::matmul at the Tensor layer.
#[allow(clippy::too_many_arguments)]
pub fn call_deformable_im2col(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize], // [batch, in_channels, height, width]
    weight_hw: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    offset_groups: usize,
    out_hw: (usize, usize),
    use_mask: bool,
    input: BufferOffset,
    offset: BufferOffset,
    mask: BufferOffset, // Can be dummy if use_mask is false
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::DeformConv2d, name)?;

    let (batch_sz, n_in_channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
    let (weight_h, weight_w) = weight_hw;
    let (out_h, out_w) = out_hw;

    let dst_el = n_in_channels * out_h * out_w * batch_sz;

    let encoder = ep.encoder();
    let encoder: &crate::ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);

    // Parameters passed individually following Candle's set_params! convention
    set_params!(
        encoder,
        (
            &input,
            &offset,
            &mask,
            output,
            height as u32,
            width as u32,
            weight_h as u32,
            weight_w as u32,
            padding.0 as u32,
            padding.1 as u32,
            stride.0 as u32,
            stride.1 as u32,
            dilation.0 as u32,
            dilation.1 as u32,
            batch_sz as u32,
            n_in_channels as u32,
            offset_groups as u32,
            out_h as u32,
            out_w as u32,
            use_mask
        )
    );

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(offset.buffer, MTLResourceUsage::Read);
    if use_mask {
        encoder.use_resource(mask.buffer, MTLResourceUsage::Read);
    }
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
