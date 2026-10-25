//! Slang dequantize dispatch: one work item per quantized block, reads the raw
//! quantized bytes from `g_Src` and writes the f32 values to `g_Dst`.
//! `params` is pushed through the pipeline's push constant block.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::push_constants::DequantParams;
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

/// Records a Slang dequantize dispatch onto `cbb`: dequantizes `num_blocks`
/// quantized blocks from `src` (raw bytes) into `dst` (f32). The shader is
/// selected by `source`/`name` (one per `GgmlDType`). `params` holds
/// `[num_blocks]`.
pub fn call_dequantize_slang(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    src: &Subbuffer<[u8]>,
    dst: &Subbuffer<[f32]>,
    params: &DequantParams,
    num_blocks: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let src_info = buf_info(src);
    let dst_info = buf_info(dst);
    
    let writes = vec![
        WriteDescriptorSet::buffer(0, &src_info),
        WriteDescriptorSet::buffer(1, &dst_info),
    ];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = (num_blocks as u64 + 255) / 256;
    unsafe { cbb.dispatch([workgroups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
