//! Slang comparison-op dispatch (Eq/Ne/Lt/Le/Gt/Ge -> u8 output).
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

/// Records a Slang comparison dispatch onto `cbb`:
/// `output[i] = (pred(input[i], rhs[rhs_offset(i)])) ? 1 : 0`. `input` and
/// `rhs` share the element type `T` (the dtype's GPU storage type); `output`
/// is always `u8`. `params` must hold `[total, lhs_ndim, rhs_ndim, l0..l3, r0..r3]`.
pub fn call_cmp_slang<T>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    input: &Subbuffer<[T]>,
    rhs: &Subbuffer<[T]>,
    output: &Subbuffer<[u8]>,
    params: &Subbuffer<[u64]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let input_info = DescriptorBufferInfo {
        buffer: Some(input.buffer()),
        offset: input.offset(),
        range: Some(input.size()),
        ..Default::default()
    };
    let rhs_info = DescriptorBufferInfo {
        buffer: Some(rhs.buffer()),
        offset: rhs.offset(),
        range: Some(rhs.size()),
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
    let set = kernels.descriptor_set(&entry.set_layout, &[input_info, rhs_info, output_info, params_info])?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = (total as u64 + 255) / 256;
    unsafe { cbb.dispatch([workgroups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
