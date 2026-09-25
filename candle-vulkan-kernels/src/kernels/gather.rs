//! Axis-0 gather for f32 tensors with u32 row indices.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

const WORKGROUP_SIZE: usize = 256;

/// Records an axis-0 gather dispatch: `out[i, :] = emb[ids[i], :]`, where
/// `emb` has `dim` columns. One workgroup item gathers one row.
pub fn call_gather_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    ids: &Subbuffer<[u32]>,
    emb: &Subbuffer<[f32]>,
    output: &Subbuffer<[f32]>,
    rows: usize,
    dim: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Gather, KernelName::GatherF32)?;
        let emb_info = DescriptorBufferInfo {
        buffer: Some(emb.buffer()),
        offset: emb.offset(),
        range: Some(emb.size()),
        ..Default::default()
    };
    let ids_info = DescriptorBufferInfo {
        buffer: Some(ids.buffer()),
        offset: ids.offset(),
        range: Some(ids.size()),
        ..Default::default()
    };
    let output_info = DescriptorBufferInfo {
        buffer: Some(output.buffer()),
        offset: output.offset(),
        range: Some(output.size()),
        ..Default::default()
    };
    let writes = vec![
        WriteDescriptorSet::buffer(0, &ids_info),
        WriteDescriptorSet::buffer(1, &emb_info),
        WriteDescriptorSet::buffer(2, &output_info),
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
    cbb.push_constants(entry.layout.clone(), 0, [rows as u32, dim as u32])
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = ((rows as u64 + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64)) as u32;
    unsafe { cbb.dispatch([workgroups, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
