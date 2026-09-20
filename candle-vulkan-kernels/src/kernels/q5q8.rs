//! Q8_0, Q5_0 and Q5_K dequant dispatches (raw quantized bytes -> f32).
//! Used by the prefill path (dequant + f32 GEMM) and by the m == 1 decode
//! path (dequant + f32 GEMV).
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

/// Records a Q8_0 dequant dispatch: `w` is raw Q8_0 bytes holding
/// `elem_count` elements (34 bytes per 32 elements), `output` has
/// `elem_count` f32 elements.
pub fn call_q80_dequant_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    w: &Subbuffer<[u8]>,
    output: &Subbuffer<[f32]>,
    elem_count: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Q5q8, KernelName::Q80DequantF32)?;
    let writes = vec![
        WriteDescriptorSet::buffer(1, w.clone()),
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
    cbb.push_constants(entry.layout.clone(), 0, [elem_count as u32, 0u32])
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = (elem_count + 255) / 256;
    unsafe { cbb.dispatch([workgroups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}

/// Records a Q5_0 dequant dispatch: `w` is raw Q5_0 bytes holding
/// `elem_count` elements (22 bytes per 32 elements), `output` has
/// `elem_count` f32 elements.
pub fn call_q50_dequant_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    w: &Subbuffer<[u8]>,
    output: &Subbuffer<[f32]>,
    elem_count: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Q5q8, KernelName::Q50DequantF32)?;
    let writes = vec![
        WriteDescriptorSet::buffer(1, w.clone()),
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
    cbb.push_constants(entry.layout.clone(), 0, [elem_count as u32, 0u32])
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = (elem_count + 255) / 256;
    unsafe { cbb.dispatch([workgroups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}

/// Records a Q5_K dequant dispatch: `w` is raw Q5_K bytes holding
/// `elem_count` elements (176 bytes per 256 elements), `output` has
/// `elem_count` f32 elements.
pub fn call_q5k_dequant_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    w: &Subbuffer<[u8]>,
    output: &Subbuffer<[f32]>,
    elem_count: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Q5q8, KernelName::Q5KDequantF32)?;
    let writes = vec![
        WriteDescriptorSet::buffer(1, w.clone()),
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
    cbb.push_constants(entry.layout.clone(), 0, [elem_count as u32, 0u32])
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = (elem_count + 255) / 256;
    unsafe { cbb.dispatch([workgroups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
