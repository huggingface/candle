//! Slang last-axis min/argmin/argmax dispatch (float input; float or u32 out).
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
/// Records a last-axis reduction onto `cbb` for element type `T`. Provide
/// exactly one of `out_f` (min/sum/max) or `out_i` (argmin/argmax).
/// `params` must hold `[rows, cols]`.
pub fn call_reduce_slang<T>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    input: &Subbuffer<[T]>,
    out_f: Option<&Subbuffer<[T]>>,
    out_i: Option<&Subbuffer<[u32]>>,
    params: &Subbuffer<[u64]>,
    rows: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let in_info = buf_info(input);
    let params_info = buf_info(params);
    let out_f_info = out_f.map(buf_info);
    let out_i_info = out_i.map(buf_info);
    let mut infos = vec![in_info];
    if let Some(info) = out_f_info {
        infos.push(info);
    }
    if let Some(info) = out_i_info {
        infos.push(info);
    }
    infos.push(params_info);
    let set = kernels.descriptor_set(&entry.set_layout, &infos)?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    unsafe { cbb.dispatch([rows as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
