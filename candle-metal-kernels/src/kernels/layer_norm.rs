use crate::linear_split;
use crate::metal::CommandBuffer;
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

/// Wrapper function compatible with ops.rs calling convention
#[allow(clippy::too_many_arguments)]
pub fn call_layer_norm_ops(
    device: &Device,
    command_buffer: &CommandBuffer,
    kernels: &Kernels,
    name: &'static str,
    elem_count: usize,
    last_dim: usize,
    eps: f32,
    inp: &Buffer,
    inp_offset: usize,
    weight: &Buffer,
    weight_offset: usize,
    bias: &Buffer,
    bias_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let num_elements = elem_count / last_dim;
    let hidden_size = last_dim;

    let input_bo = BufferOffset {
        buffer: inp,
        offset_in_bytes: inp_offset,
    };
    let weight_bo = BufferOffset {
        buffer: weight,
        offset_in_bytes: weight_offset,
    };
    let bias_bo = BufferOffset {
        buffer: bias,
        offset_in_bytes: bias_offset,
    };

    // Use basic kernel for now - can be optimized later
    call_layer_norm(
        device,
        command_buffer,
        kernels,
        name,
        num_elements,
        hidden_size,
        eps,
        input_bo,
        weight_bo,
        bias_bo,
        output,
    )
}

/// Wrapper function for RMS norm compatible with ops.rs calling convention
#[allow(clippy::too_many_arguments)]
pub fn call_rms_norm_ops(
    _device: &Device,
    _command_buffer: &CommandBuffer,
    _kernels: &Kernels,
    _name: &'static str,
    _elem_count: usize,
    _last_dim: usize,
    _eps: f32,
    _inp: &Buffer,
    _inp_offset: usize,
    _weight: &Buffer,
    _weight_offset: usize,
    _output: &Buffer,
) -> Result<(), MetalKernelError> {
    // RMS norm implementation would go here
    // For now, return unimplemented since we're focusing on layer norm
    Err(MetalKernelError::LoadFunctionError(
        "RMS norm not yet implemented for new Metal kernels".to_string(),
    ))
}

/// Call basic layer normalization kernel (F32 or F16)
///
/// # Arguments
/// * `kernel_name` - Name of Metal kernel: "layer_norm_f32" or "layer_norm_f16"
/// * `num_elements` - Number of sequence positions (batch * seq_length)
/// * `hidden_size` - Size of the last dimension to normalize over
/// * `eps` - Epsilon for numerical stability (typically 1e-5 to 1e-12)
#[allow(clippy::too_many_arguments)]
pub fn call_layer_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    num_elements: usize,
    hidden_size: usize,
    eps: f32,
    input: BufferOffset,
    weight: BufferOffset,
    bias: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::LayerNorm, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            &input,
            output,
            &weight,
            &bias,
            eps,
            hidden_size as u32,
            num_elements as u32
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_elements);
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.use_resource(weight.buffer, MTLResourceUsage::Read);
    encoder.use_resource(bias.buffer, MTLResourceUsage::Read);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Call optimized layer normalization kernel with threadgroup memory
///
/// # Arguments
/// * `kernel_name` - Name of Metal kernel: "layer_norm_f32_optimized" or "layer_norm_f16_optimized"
/// * `num_elements` - Number of sequence positions (batch * seq_length)
/// * `hidden_size` - Size of the last dimension to normalize over
/// * `eps` - Epsilon for numerical stability
///
/// Note: Uses threadgroup memory for parallel reduction (2-3x faster than basic version)
#[allow(clippy::too_many_arguments)]
pub fn call_layer_norm_optimized(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    num_elements: usize,
    hidden_size: usize,
    eps: f32,
    input: BufferOffset,
    weight: BufferOffset,
    bias: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::LayerNorm, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            &input,
            output,
            &weight,
            &bias,
            eps,
            hidden_size as u32,
            num_elements as u32
        )
    );

    // Configure threadgroup size for optimized kernel
    // Each threadgroup processes one sequence position
    let thread_group_count = MTLSize {
        width: num_elements,
        height: 1,
        depth: 1,
    };

    // Use up to 256 threads per threadgroup for parallel reduction
    let threads_per_group = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        std::cmp::min(256, hidden_size.next_power_of_two()),
    );

    let thread_group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.use_resource(weight.buffer, MTLResourceUsage::Read);
    encoder.use_resource(bias.buffer, MTLResourceUsage::Read);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Call strided layer normalization kernel for non-contiguous tensors
///
/// # Arguments
/// * `kernel_name` - Name of Metal kernel: "layer_norm_f32_strided" or "layer_norm_f16_strided"
/// * `shape` - Shape of the input tensor
/// * `strides` - Strides for each dimension
/// * `hidden_size` - Size of the last dimension to normalize over
/// * `eps` - Epsilon for numerical stability
#[allow(clippy::too_many_arguments)]
pub fn call_layer_norm_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: &[usize],
    hidden_size: usize,
    eps: f32,
    input: BufferOffset,
    weight: BufferOffset,
    bias: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::LayerNorm, kernel_name)?;
    let num_elements: usize = shape[..shape.len() - 1].iter().product();
    let num_dims = shape.len();

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            &input,
            output,
            &weight,
            &bias,
            eps,
            hidden_size as u32,
            num_elements as u32,
            num_dims,
            shape,
            strides
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_elements);
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.use_resource(weight.buffer, MTLResourceUsage::Read);
    encoder.use_resource(bias.buffer, MTLResourceUsage::Read);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
