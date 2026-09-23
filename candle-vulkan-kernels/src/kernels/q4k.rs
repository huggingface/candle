//! Q4_K quantized ops: GEMV for the m == 1 decode case and full dequant
//! (raw Q4_K bytes -> f32) for prefill and load-time paths.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
/// Records a Q4_K GEMV dispatch: `a` has `k` elements (the single input
/// row, f32), `w` is the raw Q4_K weight of shape `(n, k)` (row-major,
/// 144 bytes per 256 elements), `output` has `n` elements.
pub fn call_q4k_qmatvec_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    a: &Subbuffer<[f32]>,
    w: &Subbuffer<[u8]>,
    output: &Subbuffer<[f32]>,
    k: usize,
    n: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Q4k, KernelName::Q4kQmatvecF32)?;
    let writes = vec![
        WriteDescriptorSet::buffer(0, a.clone()),
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
    cbb.push_constants(entry.layout.clone(), 0, [k as u32, n as u32])
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    // main_qmatvec computes one output row per workgroup of 256 threads.
    unsafe { cbb.dispatch([n as u32 * 256, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
/// Records a Q4_K dequant dispatch: `w` is raw Q4_K bytes holding
/// `elem_count` elements, `output` has `elem_count` f32 elements.
pub fn call_q4k_dequant_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    w: &Subbuffer<[u8]>,
    output: &Subbuffer<[f32]>,
    elem_count: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Q4k, KernelName::Q4kDequantF32)?;
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

/// Records a Q6_K dequant dispatch: `w` is raw Q6_K bytes holding
/// `elem_count` elements, `output` has `elem_count` f32 elements.
pub fn call_q6k_dequant_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    w: &Subbuffer<[u8]>,
    output: &Subbuffer<[f32]>,
    elem_count: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Q4k, KernelName::Q6kDequantF32)?;
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
