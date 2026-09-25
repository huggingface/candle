//! Slang scatter dispatch (f32 data, u32 ids).
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

/// Records a scatter dispatch onto `cbb` (see scatter.slang for the params).
pub fn call_scatter_slang_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    name: KernelName,
    src: &Subbuffer<[f32]>,
    dst: &Subbuffer<[f32]>,
    ids: &Subbuffer<[u32]>,
    params: &Subbuffer<[f32]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::ScatterSlang, name)?;
    let src_info = DescriptorBufferInfo {
        buffer: Some(src.buffer()),
        offset: src.offset(),
        range: Some(src.size()),
        ..Default::default()
    };
    let dst_info = DescriptorBufferInfo {
        buffer: Some(dst.buffer()),
        offset: dst.offset(),
        range: Some(dst.size()),
        ..Default::default()
    };
    let ids_info = DescriptorBufferInfo {
        buffer: Some(ids.buffer()),
        offset: ids.offset(),
        range: Some(ids.size()),
        ..Default::default()
    };
    let params_info = DescriptorBufferInfo {
        buffer: Some(params.buffer()),
        offset: params.offset(),
        range: Some(params.size()),
        ..Default::default()
    };
    let writes = vec![
        WriteDescriptorSet::buffer(0, &src_info),
        WriteDescriptorSet::buffer(1, &dst_info),
        WriteDescriptorSet::buffer(2, &ids_info),
        WriteDescriptorSet::buffer(3, &params_info),
    ];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
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
