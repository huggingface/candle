//! Slang last-axis min/argmin/argmax dispatch (f32 input; f32 or u32 out).
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

/// Records a last-axis reduction onto `cbb`. Provide exactly one of
/// `out_f` (min) or `out_i` (argmin/argmax).
/// `params` must hold `[rows, cols]`.
pub fn call_reduce_slang_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    name: KernelName,
    input: &Subbuffer<[f32]>,
    out_f: Option<&Subbuffer<[f32]>>,
    out_i: Option<&Subbuffer<[u32]>>,
    params: &Subbuffer<[f32]>,
    rows: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::ReduceSlang, name)?;
    let in_info = DescriptorBufferInfo {
        buffer: Some(input.buffer()),
        offset: input.offset(),
        range: Some(input.size()),
        ..Default::default()
    };
    let params_info = DescriptorBufferInfo {
        buffer: Some(params.buffer()),
        offset: params.offset(),
        range: Some(params.size()),
        ..Default::default()
    };
    let out_f_info = out_f.map(|of| DescriptorBufferInfo {
        buffer: Some(of.buffer()),
        offset: of.offset(),
        range: Some(of.size()),
        ..Default::default()
    });
    let out_i_info = out_i.map(|oi| DescriptorBufferInfo {
        buffer: Some(oi.buffer()),
        offset: oi.offset(),
        range: Some(oi.size()),
        ..Default::default()
    });
    let mut writes = vec![WriteDescriptorSet::buffer(0, &in_info)];
    if let Some(info) = &out_f_info {
        writes.push(WriteDescriptorSet::buffer(1, info));
    }
    if let Some(info) = &out_i_info {
        writes.push(WriteDescriptorSet::buffer(2, info));
    }
    writes.push(WriteDescriptorSet::buffer(3, &params_info));
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    unsafe { cbb.dispatch([rows as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
