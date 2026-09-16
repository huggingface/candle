//! `dst[i] = src[i] * mul + add` over f32 storage buffers.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

const WORKGROUP_SIZE: usize = 256;

/// Records an `affine_f32` dispatch onto `cbb`.
pub fn call_affine_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    input: &Subbuffer<[f32]>,
    output: &Subbuffer<[f32]>,
    mul: f32,
    add: f32,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Affine, KernelName::AffineF32)?;
    let size = input.len();
    let writes = vec![
        WriteDescriptorSet::buffer(0, input.clone()),
        WriteDescriptorSet::buffer(1, output.clone()),
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
    cbb.bind_descriptor_sets(PipelineBindPoint::COMPUTE, entry.layout.clone(), 0, [set])
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, [size as u32, mul.to_bits(), add.to_bits()])
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = size.div_ceil(WORKGROUP_SIZE) as u32;
    unsafe { cbb.dispatch([workgroups, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
