//! Shared dispatch for single-output-buffer test fill kernels.
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

/// Loads `source`/`name`, binds `out` at descriptor set binding 0, and
/// dispatches a single-output-buffer kernel using the given workgroup size.
pub fn dispatch_single_buffer<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    workgroup_size: usize,
    out: &Subbuffer<[T]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let out_info = DescriptorBufferInfo {
        buffer: Some(out.buffer()),
        offset: out.offset(),
        range: Some(out.size()),
        ..Default::default()
    };
    let writes = [WriteDescriptorSet::buffer(0, &out_info)];
    let set = DescriptorSet::new(
        kernels.dss_alloc(),
        &entry.set_layout,
        &writes,
        &[],
    )
    .map_err(|e| match e {
        vulkano::Validated::Error(e) => VulkanKernelError::DescriptorSet(e.to_string()),
        vulkano::Validated::ValidationError(e) => VulkanKernelError::DescriptorSet(format!(
            "validation error: {} (requires: {}; vuids: {:?})",
            e.problem, e.requires_one_of, e.vuids
        )),
    })?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups =
        ((total as u64 + (workgroup_size as u64) - 1) / (workgroup_size as u64)) as u32;
    unsafe { cbb.dispatch([workgroups, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
