//! Sum reduction over the last (contiguous) axis for f32 tensors.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

/// Records a `reduce_sum_f32` dispatch: one workgroup per row of the input.
pub fn call_reduce_sum_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    input: &Subbuffer<[f32]>,
    output: &Subbuffer<[f32]>,
    rows: usize,
    cols: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Reduce, KernelName::ReduceSumF32)?;
    let writes = vec![
        WriteDescriptorSet::buffer(0, input.clone()),
        WriteDescriptorSet::buffer(1, output.clone()),
    ];
    let set = DescriptorSet::new(
        kernels.dss_alloc().clone(),
        entry.set_layout.clone(),
        writes,
        Vec::new(),
    )
    .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;

    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::COMPUTE, entry.layout.clone(), 0, [set])
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, [rows as u32, cols as u32])
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = rows as u32;
    unsafe { cbb.dispatch([workgroups, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
