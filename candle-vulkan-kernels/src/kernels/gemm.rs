//! Slang tiled f32 GEMM dispatch: `out[b, m, n] = lhs[b, m, k] @ rhs[b, k, n]`.
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;
const TILE: usize = 16;
/// Records a tiled GEMM dispatch onto `cbb`. `params` must hold
/// `[batch, m, n, k, rhs_transposed]`; the dispatch is
/// `(ceil(m/16), ceil(n/16), batch)` workgroups of 16x16.
pub fn call_gemm_slang_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    lhs: &Subbuffer<[f32]>,
    rhs: &Subbuffer<[f32]>,
    output: &Subbuffer<[f32]>,
    params: &Subbuffer<[f32]>,
    batch: usize,
    m: usize,
    n: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Gemm, KernelName::GemmF32)?;
    let mut infos = Vec::new();
    for buf in [lhs, rhs, output, params] {
        infos.push(DescriptorBufferInfo {
            buffer: Some(buf.buffer()),
            offset: buf.offset(),
            range: Some(buf.size()),
            ..Default::default()
        });
    }
    let writes = vec![
        WriteDescriptorSet::buffer(0, &infos[0]),
        WriteDescriptorSet::buffer(1, &infos[1]),
        WriteDescriptorSet::buffer(2, &infos[2]),
        WriteDescriptorSet::buffer(3, &infos[3]),
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
    unsafe {
        cbb.dispatch([
            ((m + TILE - 1) / TILE) as u32,
            ((n + TILE - 1) / TILE) as u32,
            batch as u32,
        ])
    }
    .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
