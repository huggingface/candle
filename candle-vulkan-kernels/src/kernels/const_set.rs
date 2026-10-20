//! Constant fill dispatch. Parameters (`count`, `value`) travel through
//! the pipeline's push constant block.
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::push_constants::ConstSetParams;
use crate::source::Source;
/// Records a constant-fill dispatch (`main`) onto `cbb`: the first `total`
/// elements of `out` get `value` written to them. `params` (`count`,
/// `value`) is pushed through the pipeline's push constant block.
pub fn call_const_set_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[f32]>,
    params: &ConstSetParams,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::ConstSet, KernelName::ConstSetF32)?;
    let out_info = DescriptorBufferInfo {
        buffer: Some(out.buffer()),
        offset: out.offset(),
        range: Some(out.size()),
        ..Default::default()
    };
    let writes = [WriteDescriptorSet::buffer(0, &out_info)];
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
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = (total as u64 + 255) / 256;
    unsafe { cbb.dispatch([workgroups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}

/// Records a constant-fill dispatch (`main`) onto `cbb`: the first `total`
/// elements of `out` get `value` written to them. `params` (`count`,
/// `value`) is pushed through the pipeline's push constant block. `out`
/// must be a f16 storage buffer.
pub fn call_const_set_f16<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
    params: &ConstSetParams,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::ConstSetF16, KernelName::ConstSetF16)?;
    let out_info = DescriptorBufferInfo {
        buffer: Some(out.buffer()),
        offset: out.offset(),
        range: Some(out.size()),
        ..Default::default()
    };
    let writes = [WriteDescriptorSet::buffer(0, &out_info)];
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
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = (total as u64 + 255) / 256;
    unsafe { cbb.dispatch([workgroups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}

/// Records a constant-fill dispatch (`main`) onto `cbb`: the first `total`
/// elements of `out` get `value` written to them. `params` (`count`,
/// `value`) is pushed through the pipeline's push constant block. `out`
/// must be a bf16 storage buffer.
pub fn call_const_set_bf16<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
    params: &ConstSetParams,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::ConstSetBf16, KernelName::ConstSetBf16)?;
    let out_info = DescriptorBufferInfo {
        buffer: Some(out.buffer()),
        offset: out.offset(),
        range: Some(out.size()),
        ..Default::default()
    };
    let writes = [WriteDescriptorSet::buffer(0, &out_info)];
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
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = (total as u64 + 255) / 256;
    unsafe { cbb.dispatch([workgroups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
