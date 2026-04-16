use crate::linear_split;
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::MTLResourceUsage;

#[allow(clippy::too_many_arguments)]
pub fn call_index_select(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    ids_size: usize,
    dim: usize,
    contiguous: bool,
    src_dims: &[usize],
    src_strides: &[usize],
    input: BufferOffset,
    ids: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let src_dim_size = shape[dim];
    let dst_el = ids_size * left_size * right_size;

    let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            dst_el,
            left_size,
            src_dim_size,
            right_size,
            ids_size,
            contiguous,
            src_dims,
            src_strides,
            &input,
            &ids,
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(ids.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_gather(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    ids_size: usize,
    dim: usize,
    input: BufferOffset,
    ids: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let src_dim_size = shape[dim];
    let dst_el = ids_size * left_size * right_size;

    let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            dst_el,
            left_size,
            src_dim_size,
            right_size,
            ids_size,
            &input,
            &ids,
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(ids.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_scatter(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    src_shape: &[usize],
    dst_shape: &[usize],
    dim: usize,
    input: BufferOffset,
    ids: BufferOffset,
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    let left_size: usize = src_shape[..dim].iter().product();
    let right_size: usize = src_shape[dim + 1..].iter().product();
    let src_dim_size = src_shape[dim];
    let dst_el = left_size * right_size;
    let dst_dim_size = dst_shape[dim];

    let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            dst_el,
            left_size,
            src_dim_size,
            right_size,
            dst_dim_size,
            &input,
            &ids,
            &output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(ids.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output.buffer, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_index_add(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    src_shape: &[usize],
    dst_shape: &[usize],
    ids_shape: &[usize],
    dim: usize,
    input: BufferOffset,
    ids: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let left_size: usize = src_shape[..dim].iter().product();
    let right_size: usize = src_shape[dim + 1..].iter().product();
    let src_dim_size = src_shape[dim];
    let dst_el = left_size * right_size;
    let dst_dim_size = dst_shape[dim];
    let ids_dim_size = ids_shape[0];

    let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            dst_el,
            left_size,
            src_dim_size,
            right_size,
            dst_dim_size,
            ids_dim_size,
            &input,
            &ids,
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(ids.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
