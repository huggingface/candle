//! f32 elementwise dispatches: broadcast binary ops and unary ops.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

const WORKGROUP_SIZE: usize = 256;

/// Records a broadcast binary elementwise dispatch (`main_add`/`main_sub`/
/// `main_mul`/`main_div`) onto `cbb`. `lhs_dims` and `rhs_dims` carry the
/// broadcast shape of the inputs: the rhs is right-aligned against the
/// lhs, with a dim of 1 (or an empty rhs, the scalar case) selecting the
/// corresponding index 0.
pub fn call_elem_binary_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    name: KernelName,
    input: &Subbuffer<[f32]>,
    rhs: &Subbuffer<[f32]>,
    output: &Subbuffer<[f32]>,
    lhs_dims: &[usize],
    rhs_dims: &[usize],
) -> Result<(), VulkanKernelError> {
    if lhs_dims.len() > 4 || rhs_dims.len() > 4 {
        return Err(VulkanKernelError::Message(
            "elementwise binary: more than 4 dims".into(),
        ));
    }
    let total = input.len();
    let mut ldims = [0u32; 4];
    for (slot, d) in ldims.iter_mut().zip(lhs_dims) {
        *slot = *d as u32;
    }
    let mut rdims = [0u32; 4];
    for (slot, d) in rdims.iter_mut().zip(rhs_dims) {
        *slot = *d as u32;
    }
    let entry = kernels.load_entry(Source::Elementwise, name)?;
    let writes = vec![
        WriteDescriptorSet::buffer(0, input.clone()),
        WriteDescriptorSet::buffer(1, rhs.clone()),
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
    cbb.push_constants(
        entry.layout.clone(),
        0,
        [
            total as u32,
            lhs_dims.len() as u32,
            rhs_dims.len() as u32,
            ldims[0],
            ldims[1],
            ldims[2],
            ldims[3],
            rdims[0],
            rdims[1],
            rdims[2],
            rdims[3],
        ],
    )
    .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = ((total + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64)) as u32;
    unsafe { cbb.dispatch([workgroups, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}

/// Records a unary elementwise dispatch (`main_sigmoid`/`main_silu`/
/// `main_exp`/`main_sqrt`/`main_sin`/`main_cos`/`main_neg`) onto `cbb`.
pub fn call_elem_unary_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    name: KernelName,
    input: &Subbuffer<[f32]>,
    output: &Subbuffer<[f32]>,
) -> Result<(), VulkanKernelError> {
    let total = input.len();
    let entry = kernels.load_entry(Source::Elementwise, name)?;
    let writes = vec![
        WriteDescriptorSet::buffer(0, input.clone()),
        // The shader module is shared between the unary and binary entry
        // points: `out_buf` is binding 2, `rhs_buf` (binding 1) is unused
        // by the unary ops and left unwritten.
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
    let mut pc = [0u32; 11];
    pc[0] = total as u32;
    // The layout's push-constant range spans the whole shared block
    // (44 bytes) even though the unary entry point only reads `total`.
    cbb.push_constants(entry.layout.clone(), 0, pc)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = ((total + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64)) as u32;
    unsafe { cbb.dispatch([workgroups, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
