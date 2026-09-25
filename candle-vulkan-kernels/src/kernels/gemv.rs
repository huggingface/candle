//! GEMV for the m=1 matmul case: `(1, k) @ (k, n) -> (1, n)`.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels, GEMV_T_TILE_N};
use crate::source::Source;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

/// Records a `gemv_f32` dispatch: `a` has `k` elements, `w` is the
/// row-major `(k, n)` weight, `output` has `n` elements.
pub fn call_gemv_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    a: &Subbuffer<[f32]>,
    w: &Subbuffer<[f32]>,
    output: &Subbuffer<[f32]>,
    k: usize,
    n: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Gemv, KernelName::GemvF32)?;
    let a_info = DescriptorBufferInfo {
        buffer: Some(a.buffer()),
        offset: a.offset(),
        range: Some(a.size()),
        ..Default::default()
    };
    let output_info = DescriptorBufferInfo {
        buffer: Some(output.buffer()),
        offset: output.offset(),
        range: Some(output.size()),
        ..Default::default()
    };
    let w_info = DescriptorBufferInfo {
        buffer: Some(w.buffer()),
        offset: w.offset(),
        range: Some(w.size()),
        ..Default::default()
    };
    let writes = vec![
        WriteDescriptorSet::buffer(0, &a_info),
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
    cbb.push_constants(entry.layout.clone(), 0, [k as u32, n as u32])
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = (n + 255) / 256;
    unsafe { cbb.dispatch([workgroups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}

/// Records a transposed GEMV dispatch (one workgroup per 4 output rows,
/// full-k tile staged in workgroup memory).
pub fn call_gemv_t_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    a: &Subbuffer<[f32]>,
    w: &Subbuffer<[f32]>,
    output: &Subbuffer<[f32]>,
    n: usize,
    k: usize,
    w_stride: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::GemvT, KernelName::GemvTF32)?;
    let a_info = DescriptorBufferInfo {
        buffer: Some(a.buffer()),
        offset: a.offset(),
        range: Some(a.size()),
        ..Default::default()
    };
    let output_info = DescriptorBufferInfo {
        buffer: Some(output.buffer()),
        offset: output.offset(),
        range: Some(output.size()),
        ..Default::default()
    };
    let w_info = DescriptorBufferInfo {
        buffer: Some(w.buffer()),
        offset: w.offset(),
        range: Some(w.size()),
        ..Default::default()
    };
    let writes = vec![
        WriteDescriptorSet::buffer(0, &a_info),
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
    cbb.push_constants(
        entry.layout.clone(),
        0,
        [n as u32, k as u32, w_stride as u32],
    )
    .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let grid_x = (n as u32 + GEMV_T_TILE_N as u32 - 1) / GEMV_T_TILE_N as u32;
    unsafe { cbb.dispatch([grid_x, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
