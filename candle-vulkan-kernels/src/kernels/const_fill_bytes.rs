//! Byte-granular constant fill: stamps a 4-byte pattern over a byte region.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::push_constants::ConstFillBytesParams;
use crate::source::Source;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

fn buf_info(buf: &Subbuffer<[u8]>) -> DescriptorBufferInfo<'_> {
    DescriptorBufferInfo {
        buffer: Some(buf.buffer()),
        offset: buf.offset(),
        range: Some(buf.size()),
        ..Default::default()
    }
}

/// Records a byte-granular constant fill onto `cbb`: every byte of `dst`
/// gets the corresponding byte of the 4-byte `pattern` (repeated). `params`
/// is pushed through the pipeline's push constant block; the dispatch is
/// `(ceil(byte_len/256), 1, 1)` workgroups of 256 threads.
pub fn call_const_fill_bytes(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    dst: &Subbuffer<[u8]>,
    params: &ConstFillBytesParams,
    byte_len: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::ConstFillBytes, KernelName::ConstFillBytes)?;
    let info0 = buf_info(dst);

    let writes = vec![WriteDescriptorSet::buffer(0, &info0)];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = (byte_len as u64 + 255) / 256;
    unsafe { cbb.dispatch([workgroups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
