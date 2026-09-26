//! Slang unary-op dispatch (ops not covered by the elementwise WGSL).
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

/// Records a Slang unary-op dispatch onto `cbb`: `output[i] = f(input[i])`
/// for `i < total`. `input` and `output` share the element type `T` (the
/// dtype's GPU storage type). `params` must hold `[total as f32]`.
pub fn call_unary_slang<T>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    input: &Subbuffer<[T]>,
    output: &Subbuffer<[T]>,
    params: &Subbuffer<[f32]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let input_info = DescriptorBufferInfo {
        buffer: Some(input.buffer()),
        offset: input.offset(),
        range: Some(input.size()),
        ..Default::default()
    };
    let output_info = DescriptorBufferInfo {
        buffer: Some(output.buffer()),
        offset: output.offset(),
        range: Some(output.size()),
        ..Default::default()
    };
    let params_info = DescriptorBufferInfo {
        buffer: Some(params.buffer()),
        offset: params.offset(),
        range: Some(params.size()),
        ..Default::default()
    };
    let writes = vec![
        WriteDescriptorSet::buffer(0, &input_info),
        WriteDescriptorSet::buffer(1, &output_info),
        WriteDescriptorSet::buffer(2, &params_info),
    ];
    let set = DescriptorSet::new(
        kernels.dss_alloc(),
        &entry.set_layout,
        &writes,
        &[],
    )
    .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = (total as u64 + 255) / 256;
    unsafe { cbb.dispatch([workgroups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
