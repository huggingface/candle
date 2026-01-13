use crate::linear_split;
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

#[allow(clippy::too_many_arguments)]
pub fn call_reduce_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let length = shape.iter().product::<usize>();
    let num_dims = shape.len();
    let work_per_threadgroup = length / out_length;

    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let shape: Vec<u32> = shape.iter().map(|&x| x as u32).collect();
    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape.as_slice(),
            work_per_threadgroup,
            &input,
            output
        )
    );

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: out_length,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_reduce_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let length: usize = shape.iter().product();
    let num_dims = shape.len();
    let work_per_threadgroup = length / out_length;

    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let shape: Vec<u32> = shape.iter().map(|&x| x as u32).collect();
    let strides: Vec<u32> = strides.iter().map(|&x| x as u32).collect();
    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape.as_slice(),
            strides.as_slice(),
            work_per_threadgroup,
            &input,
            output
        )
    );

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: out_length,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_last_softmax(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let work_per_threadgroup = elements;

    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (length, work_per_threadgroup, (input, input_offset), output)
    );

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rms_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_to_sum,
            (input, input_offset),
            output,
            (alpha, alpha_offset),
            eps
        )
    );
    let work_per_threadgroup = elements_to_sum;

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(alpha, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_layer_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    beta: &Buffer,
    beta_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_to_sum,
            (input, input_offset),
            output,
            (alpha, alpha_offset),
            (beta, beta_offset),
            eps
        )
    );

    let work_per_threadgroup = elements_to_sum;

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(alpha, MTLResourceUsage::Read);
    encoder.use_resource(beta, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope_i(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            bh,
            td,
            stride_b,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope_thd(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    b: usize,
    t: usize,
    h: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            b,
            t,
            h,
            d,
            stride_b,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (b * t * h * d) / 2);
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            bh,
            td,
            d,
            stride_b,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
