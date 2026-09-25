//! Fused RMSNorm over the last (contiguous) axis for f32 tensors.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
/// Records a fused `rms_norm_f32` dispatch: one workgroup per row of the
/// input, `weight` has `cols` elements and is applied elementwise.
pub fn call_rms_norm_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    input: &Subbuffer<[f32]>,
    weight: &Subbuffer<[f32]>,
    output: &Subbuffer<[f32]>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::RmsNorm, KernelName::RmsNormF32)?;
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
    let weight_info = DescriptorBufferInfo {
        buffer: Some(weight.buffer()),
        offset: weight.offset(),
        range: Some(weight.size()),
        ..Default::default()
    };
    let writes = vec![
        WriteDescriptorSet::buffer(0, &input_info),
        WriteDescriptorSet::buffer(1, &weight_info),
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
    cbb.push_constants(
        entry.layout.clone(),
        0,
        [rows as u32, cols as u32, eps.to_bits()],
    )
    .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = rows as u32;
    unsafe { cbb.dispatch([workgroups, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
