//! Slang fused single-query decode attention dispatch. Parameters travel
//! through the pipeline's push constant block.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::push_constants::DecodeAttentionParams;
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

/// Records a fused single-query decode attention dispatch: `q` is
/// `(num_heads, head_dim)` f32, `k` and `v` are `(num_kv_heads, kv_len,
/// head_dim)` f32, `dst` is `(num_heads, head_dim)` f32. `params` travels
/// through the push constant block. The dispatch is `(num_heads, 1, 1)`
/// workgroups of 32 threads; one workgroup per head.
pub fn call_decode_attention_slang(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    q: &Subbuffer<[f32]>,
    k: &Subbuffer<[f32]>,
    v: &Subbuffer<[f32]>,
    dst: &Subbuffer<[f32]>,
    params: &DecodeAttentionParams,
    num_heads: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let q_info = buf_info(q);
    let k_info = buf_info(k);
    let v_info = buf_info(v);
    let dst_info = buf_info(dst);
    let writes = vec![
        WriteDescriptorSet::buffer(0, &q_info),
        WriteDescriptorSet::buffer(1, &k_info),
        WriteDescriptorSet::buffer(2, &v_info),
        WriteDescriptorSet::buffer(3, &dst_info),
    ];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    unsafe { cbb.dispatch([1, num_heads as u32, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
