//! Slang `dst[i] = src[i] * mul + add` dispatch over dtype storage buffers.
//! Parameters travel through the pipeline's push constant block.
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::push_constants::{AffineParams, GemmParams};
use crate::source::Source;

const WORKGROUP_SIZE: usize = 256;

/// Records an affine dispatch onto `cbb`. `input` and `output` share the
/// element type `T` (the dtype's GPU storage type). `params` (`total`,
/// `mul`, `add`) is pushed through the pipeline's push constant block.
pub fn call_affine_slang<T>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    input: &Subbuffer<[T]>,
    output: &Subbuffer<[T]>,
    params: &AffineParams,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
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
    let writes = vec![
        WriteDescriptorSet::buffer(0, &input_info),
        WriteDescriptorSet::buffer(1, &output_info),
    ];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = ((total + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) as u32;
    unsafe { cbb.dispatch([workgroups, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}

/// Cooperative-matrix affine (GEMM + bias) dispatch:
/// `out[b, m, n] = lhs[b, m, k] @ rhs[b, k, n] + bias[n]` for element
/// type `T` (the dtype's GPU storage type). `params` is pushed through the pipeline's push constant
/// block; the dispatch is
/// `(ceil(m/16), ceil(n/16), batch)` workgroups of 64.
pub fn call_affine_coopmat_slang<T>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    lhs: &Subbuffer<[T]>,
    rhs: &Subbuffer<[T]>,
    bias: &Subbuffer<[T]>,
    output: &Subbuffer<[T]>,
    params: &GemmParams,
    batch: usize,
    m: usize,
    n: usize,
) -> Result<(), VulkanKernelError> {
    // RADV workaround: prime the output buffer so a readback recorded
    // after this dispatch in the same command buffer sees the kernel's
    // store (see `Kernels::prime_coopmat_output`).
    kernels.prime_coopmat_output(cbb, output)?;
    let entry = kernels.load_entry(source, name)?;
    let info0 = super::gemm::buf_info(lhs);
    let info1 = super::gemm::buf_info(rhs);
    let info2 = super::gemm::buf_info(bias);
    let info3 = super::gemm::buf_info(output);
    
    let writes = vec![
        WriteDescriptorSet::buffer(0, &info0),
        WriteDescriptorSet::buffer(1, &info1),
        WriteDescriptorSet::buffer(2, &info2),
        WriteDescriptorSet::buffer(3, &info3),
    ];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    unsafe { cbb.dispatch([((m + 15) / 16) as u32, ((n + 15) / 16) as u32, batch as u32]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
