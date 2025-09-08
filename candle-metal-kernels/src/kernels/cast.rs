use crate::linear_split;
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::MTLResourceUsage;

#[allow(clippy::too_many_arguments)]
pub fn call_cast_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Cast, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, &input, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_cast_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Cast, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let length: usize = shape.iter().product();

    set_params!(
        encoder,
        (length, shape.len(), shape, input_strides, &input, output)
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
