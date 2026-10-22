//! Slang rotary-embedding dispatch: one thread per (bh_i, i_t, i_d) pair.
//! `params` must hold `[b*h, t, d]`.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
fn buf_info<T>(buf: &Subbuffer<[T]>) -> DescriptorBufferInfo<'_> {
    DescriptorBufferInfo {
        buffer: Some(buf.buffer()),
        offset: buf.offset(),
        range: Some(buf.size()),
        ..Default::default()
    }
}
/// Records a rotary-embedding dispatch: `src` is `(b, h, t, d)` f32,
/// `cos`/`sin` are `(t, d/2)` f32, `dst` is `(b, h, t, d)` f32. `params`
/// holds `[b*h, t, d]`.
pub fn call_rope_slang(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    src: &Subbuffer<[f32]>,
    cos: &Subbuffer<[f32]>,
    sin: &Subbuffer<[f32]>,
    dst: &Subbuffer<[f32]>,
    params: &Subbuffer<[u64]>,
    n_threads: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let src_info = buf_info(src);
    let cos_info = buf_info(cos);
    let sin_info = buf_info(sin);
    let dst_info = buf_info(dst);
    let params_info = buf_info(params);
    let set = kernels.descriptor_set(&entry.set_layout, &[src_info, cos_info, sin_info, dst_info, params_info])?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    unsafe { cbb.dispatch([n_threads as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
