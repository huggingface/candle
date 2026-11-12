//! Slang last-axis min/argmin/argmax dispatch (float input; float or u32 out).
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::push_constants::ReduceParams;
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
/// exactly one of `out_f` (min/sum/max) or `out_i` (argmin/argmax). `params` is pushed
/// through the pipeline's push constant block.
pub fn call_reduce_slang<T>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    input: &Subbuffer<[T]>,
    out_f: Option<&Subbuffer<[T]>>,
    out_i: Option<&Subbuffer<[u32]>>,
    params: &ReduceParams,
    rows: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let in_info = buf_info(input);

    let out_f_info = out_f.map(buf_info);
    let out_i_info = out_i.map(buf_info);
    // Always bind all three layout slots. The unused output path (min/sum/max
    // has no u32 output; argmin/argmax has no float output) is bound to the
    // input buffer so no descriptor is left null. The unused global is
    // eliminated by Slang DCE and never dereferenced.
    let mut writes = vec![WriteDescriptorSet::buffer(0, &in_info)];
    writes.push(WriteDescriptorSet::buffer(
        1,
        out_f_info.as_ref().unwrap_or(&in_info),
    ));
    writes.push(WriteDescriptorSet::buffer(
        2,
        out_i_info.as_ref().unwrap_or(&in_info),
    ));
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    unsafe { cbb.dispatch([rows as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
