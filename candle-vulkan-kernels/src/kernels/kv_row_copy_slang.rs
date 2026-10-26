//! Slang fused KV cache row-append dispatch: copies one new key row and
//! one new value row into pre-allocated caches in a single dispatch.
//! Parameters travel through the pipeline's push constant block.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::push_constants::KvRowCopyParams;
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

/// Records the fused KV row-append dispatch: `src_k`/`src_v` each hold
/// `(heads * head_dim)` f32 values (head-major), and the row is written to
/// `dst[head * stride + row * head_dim + col]` of `dst_k`/`dst_v`.
/// `params` (`heads`, `head_dim`, `row`, `stride`) is pushed through the
/// pipeline's push constant block.
pub fn call_kv_row_copy_slang(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    src_k: &Subbuffer<[f32]>,
    src_v: &Subbuffer<[f32]>,
    dst_k: &Subbuffer<[f32]>,
    dst_v: &Subbuffer<[f32]>,
    params: &KvRowCopyParams,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let src_k_info = buf_info(src_k);
    let src_v_info = buf_info(src_v);
    let dst_k_info = buf_info(dst_k);
    let dst_v_info = buf_info(dst_v);
    let writes = vec![
        WriteDescriptorSet::buffer(0, &src_k_info),
        WriteDescriptorSet::buffer(1, &src_v_info),
        WriteDescriptorSet::buffer(2, &dst_k_info),
        WriteDescriptorSet::buffer(3, &dst_v_info),
    ];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let groups = total.div_ceil(256);
    unsafe { cbb.dispatch([groups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
