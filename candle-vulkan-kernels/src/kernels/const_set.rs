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
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
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
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
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
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
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
/// must be a u32 storage buffer.
pub fn call_const_set_u32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[u32]>,
    params: &ConstSetParams,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::ConstSetU32, KernelName::ConstSetU32)?;
    let out_info = DescriptorBufferInfo {
        buffer: Some(out.buffer()),
        offset: out.offset(),
        range: Some(out.size()),
        ..Default::default()
    };
    let writes = [WriteDescriptorSet::buffer(0, &out_info)];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
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
/// must be a i64 storage buffer.
pub fn call_const_set_i64(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[i64]>,
    params: &ConstSetParams,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::ConstSetI64, KernelName::ConstSetI64)?;
    let out_info = DescriptorBufferInfo {
        buffer: Some(out.buffer()),
        offset: out.offset(),
        range: Some(out.size()),
        ..Default::default()
    };
    let writes = [WriteDescriptorSet::buffer(0, &out_info)];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
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
/// must be a u8 storage buffer.
pub fn call_const_set_u8(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[u8]>,
    params: &ConstSetParams,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::ConstSetU8, KernelName::ConstSetU8)?;
    let out_info = DescriptorBufferInfo {
        buffer: Some(out.buffer()),
        offset: out.offset(),
        range: Some(out.size()),
        ..Default::default()
    };
    let writes = [WriteDescriptorSet::buffer(0, &out_info)];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
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
/// must be a f64 storage buffer.
pub fn call_const_set_f64(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[f64]>,
    params: &ConstSetParams,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::ConstSetF64, KernelName::ConstSetF64)?;
    let out_info = DescriptorBufferInfo {
        buffer: Some(out.buffer()),
        offset: out.offset(),
        range: Some(out.size()),
        ..Default::default()
    };
    let writes = [WriteDescriptorSet::buffer(0, &out_info)];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
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
/// must be a f8e4m3 storage buffer.
pub fn call_const_set_f8e4m3<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
    params: &ConstSetParams,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::ConstSetF8E4M3, KernelName::ConstSetF8E4M3)?;
    let out_info = DescriptorBufferInfo {
        buffer: Some(out.buffer()),
        offset: out.offset(),
        range: Some(out.size()),
        ..Default::default()
    };
    let writes = [WriteDescriptorSet::buffer(0, &out_info)];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
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
