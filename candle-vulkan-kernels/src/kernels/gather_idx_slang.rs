//! Slang indexed-copy dispatch (f32 buffers, u32 index buffer).
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

/// Records an indexed copy onto `cbb`:
/// `out[i] = src[idx[i / blen] + (i % blen)]`.
/// `params` must hold `[total, block_len]`.
pub fn call_gather_idx_slang_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    src: &Subbuffer<[f32]>,
    dst: &Subbuffer<[f32]>,
    idx: &Subbuffer<[u32]>,
    params: &Subbuffer<[f32]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::GatherIdxSlang, KernelName::GatherIdxF32)?;
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
    let idx_info = DescriptorBufferInfo {
        buffer: Some(idx.buffer()),
        offset: idx.offset(),
        range: Some(idx.size()),
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
        WriteDescriptorSet::buffer(2, &idx_info),
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
