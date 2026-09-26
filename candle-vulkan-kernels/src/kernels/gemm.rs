//! Slang tiled GEMM dispatch: `out[b, m, n] = lhs[b, m, k] @ rhs[b, k, n]`.
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;
const TILE: usize = 16;
fn buf_info<T>(buf: &Subbuffer<[T]>) -> DescriptorBufferInfo<'_> {
    DescriptorBufferInfo {
        buffer: Some(buf.buffer()),
        offset: buf.offset(),
        range: Some(buf.size()),
        ..Default::default()
    }
}
/// Records a tiled GEMM dispatch onto `cbb` for element type `T`. `params`
/// (always f32) must hold `[batch, m, n, k, rhs_transposed]`; the dispatch is
/// `(ceil(m/16), ceil(n/16), batch)` workgroups of 16x16.
pub fn call_gemm_slang<T>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    lhs: &Subbuffer<[T]>,
    rhs: &Subbuffer<[T]>,
    output: &Subbuffer<[T]>,
    params: &Subbuffer<[f32]>,
    batch: usize,
    m: usize,
    n: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let info0 = buf_info(lhs);
    let info1 = buf_info(rhs);
    let info2 = buf_info(output);
    let info3 = buf_info(params);
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
