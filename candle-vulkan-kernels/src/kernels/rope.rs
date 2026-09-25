//! Non-interleaved rotary embeddings for f32 `(b, h, t, d)` tensors.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
/// Records a `rope_f32` dispatch over `(b, h, t, d)`; `cos`/`sin` have
/// `d / 2` elements per (batch,) position and `unbatched` says whether the
/// cos/sin tensors carry the batch dimension.
pub fn call_rope_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    input: &Subbuffer<[f32]>,
    cos: &Subbuffer<[f32]>,
    sin: &Subbuffer<[f32]>,
    output: &Subbuffer<[f32]>,
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    unbatched: bool,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Rope, KernelName::RopeF32)?;
        let cos_info = DescriptorBufferInfo {
        buffer: Some(cos.buffer()),
        offset: cos.offset(),
        range: Some(cos.size()),
        ..Default::default()
    };
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
    let sin_info = DescriptorBufferInfo {
        buffer: Some(sin.buffer()),
        offset: sin.offset(),
        range: Some(sin.size()),
        ..Default::default()
    };
    let writes = vec![
        WriteDescriptorSet::buffer(0, &input_info),
        WriteDescriptorSet::buffer(1, &cos_info),
        WriteDescriptorSet::buffer(2, &sin_info),
        WriteDescriptorSet::buffer(3, &output_info),
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
    let rows = (b * h * t) as u32;
    cbb.push_constants(
        entry.layout.clone(),
        0,
        [rows, h as u32, t as u32, d as u32, u32::from(unbatched)],
    )
    .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = rows;
    let cols = (d / 2) as u32;
    unsafe { cbb.dispatch([workgroups, cols, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
