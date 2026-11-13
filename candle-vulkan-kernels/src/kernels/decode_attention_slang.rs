//! Slang two-pass flash-decode single-query decode attention dispatch.
//! Parameters travel through the pipeline's push constant block.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::push_constants::DecodeAttentionParams;
use crate::source::Source;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

/// Number of KV positions handled by one (head, chunk) workgroup of the
/// first `decode_attention` pass; must match the shader's `CHUNK`
/// constant.
pub const DECODE_ATTENTION_CHUNK: usize = 128;

fn buf_info<T>(buf: &Subbuffer<[T]>) -> DescriptorBufferInfo<'_> {
    DescriptorBufferInfo {
        buffer: Some(buf.buffer()),
        offset: buf.offset(),
        range: Some(buf.size()),
        ..Default::default()
    }
}

/// Records the two-pass flash-decode dispatch. `q` is `(num_heads,
/// head_dim)` f32, `k` and `v` are `(num_kv_heads, kv_len, head_dim)`
/// f32, `scratch` is `(num_heads * num_chunks * (2 + head_dim))` f32
/// (partials written by pass 1, read by pass 2), and `dst` is
/// `(num_heads, head_dim)` f32. `params` travels through the push
/// constant block. Both entry points of the module share the five
/// buffer bindings (q=0, k=1, v=2, scratch=3, out=4), so every call
/// writes all five. Pass 1 dispatches `[1, num_heads, num_chunks]`
/// workgroups of 32 threads (heads along Y, chunks along Z); pass 2
/// dispatches `[1, num_heads, 1]`.
#[allow(clippy::too_many_arguments)]
pub fn call_decode_attention_slang(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    q: &Subbuffer<[f32]>,
    k: &Subbuffer<[half::f16]>,
    v: &Subbuffer<[half::f16]>,
    scratch: &Subbuffer<[f32]>,
    dst: &Subbuffer<[f32]>,
    params: &DecodeAttentionParams,
    num_heads: usize,
    num_chunks: usize,
) -> Result<(), VulkanKernelError> {
    let q_info = buf_info(q);
    let k_info = buf_info(k);
    let v_info = buf_info(v);
    let scratch_info = buf_info(scratch);
    let dst_info = buf_info(dst);
    let writes = vec![
        WriteDescriptorSet::buffer(0, &q_info),
        WriteDescriptorSet::buffer(1, &k_info),
        WriteDescriptorSet::buffer(2, &v_info),
        WriteDescriptorSet::buffer(3, &scratch_info),
        WriteDescriptorSet::buffer(4, &dst_info),
    ];

    // Pass 1: (head, chunk) online-softmax partials into `scratch`.
    let entry = kernels.load_entry(Source::DecodeAttention, KernelName::DecodeAttention)?;
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    unsafe { cbb.dispatch([1, num_heads as u32, num_chunks as u32]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;

    // Pass 2: merge the per-head partials into `dst`.
    let entry = kernels.load_entry(Source::DecodeAttention, KernelName::DecodeAttentionCombine)?;
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
