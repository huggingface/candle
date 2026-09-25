//! Q8_0, Q5_0 and Q5_K dequant dispatches (raw quantized bytes -> f32).
//! Used by the prefill path (dequant + f32 GEMM) and by the m == 1 decode
//! path (dequant + f32 GEMV).
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

fn dequant_info<'a>(w: &'a Subbuffer<[u8]>, output: &'a Subbuffer<[f32]>) -> (DescriptorBufferInfo<'a>, DescriptorBufferInfo<'a>) {
    let w_info = DescriptorBufferInfo {
        buffer: Some(&w.buffer()),
        offset: w.offset(),
        range: Some(w.size()),
        ..Default::default()
    };
    let output_info = DescriptorBufferInfo {
        buffer: Some(&output.buffer()),
        offset: output.offset(),
        range: Some(output.size()),
        ..Default::default()
    };
    (w_info, output_info)
}

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
    let (w_info, output_info) = dequant_info(w, output);
    let writes = vec![
        WriteDescriptorSet::buffer(1, &w_info),
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
    let (w_info, output_info) = dequant_info(w, output);
    let entry = kernels.load_entry(Source::Q5q8, KernelName::Q50DequantF32)?;
    let writes = vec![
        WriteDescriptorSet::buffer(1, &w_info),
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
    let (w_info, output_info) = dequant_info(w, output);
    let writes = vec![
        WriteDescriptorSet::buffer(1, &w_info),
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
    cbb.push_constants(entry.layout.clone(), 0, [elem_count as u32, 0u32])
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = (elem_count + 255) / 256;
    unsafe { cbb.dispatch([workgroups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
