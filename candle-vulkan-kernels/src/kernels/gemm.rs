//! Tiled f32 GEMM dispatch: `out[b, m, n] = lhs[b, m, k] @ rhs[b, k, n]`.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

const TILE: u64 = 16;

/// Records a tiled GEMM dispatch (`main_gemm`) onto `cbb`. The rhs is
/// either stored as contiguous `(b, k, n)` or transposed contiguous
/// `(b, n, k)` (`rhs_transposed` selects the indexing in the kernel).
pub fn call_gemm_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    input: &Subbuffer<[f32]>,
    rhs: &Subbuffer<[f32]>,
    output: &Subbuffer<[f32]>,
    bsz: usize,
    m: usize,
    n: usize,
    k: usize,
    rhs_transposed: bool,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Gemm, KernelName::GemmF32)?;
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
            bsz as u32,
            m as u32,
            n as u32,
            k as u32,
            rhs_transposed as u32,
        ],
    )
    .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let m_tiles = (m as u64 + TILE - 1) / TILE;
    let n_tiles = (n as u64 + TILE - 1) / TILE;
    unsafe { cbb.dispatch([m_tiles as u32, n_tiles as u32, bsz as u32]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
