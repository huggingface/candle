//! Fused RMSNorm over the last (contiguous) axis for f32 tensors.
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;
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
    let writes = vec![
        WriteDescriptorSet::buffer(0, input.clone()),
        WriteDescriptorSet::buffer(1, weight.clone()),
        WriteDescriptorSet::buffer(2, output.clone()),
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
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, [rows as u32, cols as u32, eps.to_bits()])
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = rows as u32;
    unsafe { cbb.dispatch([workgroups, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
