//! DP4A (integer dot product) Q8_0 GEMM dispatch + activation quantization.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::push_constants::{FusedQuantGemmParams, UnaryParams};
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

/// Quantizes a single-row (k,) f32 activation to i8 (see `quantize_lhs_i8`),
/// writing the i8 bytes to `lhsq` and the per-row dequant scale to `scale`.
pub fn call_quantize_lhs_i8(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    lhs: &Subbuffer<[f32]>,
    lhsq: &Subbuffer<[u8]>,
    scale: &Subbuffer<[f32]>,
    params: &UnaryParams,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::QuantizeLhsI8, KernelName::QuantizeLhsI8F32)?;
    let i0 = buf_info(lhs);
    let i1 = buf_info(lhsq);
    let i2 = buf_info(scale);
    let writes = vec![
        WriteDescriptorSet::buffer(0, &i0),
        WriteDescriptorSet::buffer(1, &i1),
        WriteDescriptorSet::buffer(2, &i2),
    ];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    unsafe { cbb.dispatch([1, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}

/// Records the DP4A Q8_0 decode GEMM `out[n] = lhsq[k] . W[n, k]` for a
/// single-row (m == 1) quantized activation. `w` holds raw Q8_0 bytes,
/// `scale` the activation's dequant scale, `out` the f32 result. Dispatch is
/// (1, ceil(n/32), 1) workgroups of 32 threads.
pub fn call_fused_q8_0_gemm_dp4a(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    lhsq: &Subbuffer<[u8]>,
    w: &Subbuffer<[u8]>,
    scale: &Subbuffer<[f32]>,
    out: &Subbuffer<[f32]>,
    params: &FusedQuantGemmParams,
    n: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::FusedQ80GemmDp4a, KernelName::FusedQ80GemmDp4aF32)?;
    let i0 = buf_info(lhsq);
    let i1 = buf_info(w);
    let i2 = buf_info(scale);
    let i3 = buf_info(out);
    let writes = vec![
        WriteDescriptorSet::buffer(0, &i0),
        WriteDescriptorSet::buffer(1, &i1),
        WriteDescriptorSet::buffer(2, &i2),
        WriteDescriptorSet::buffer(3, &i3),
    ];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    unsafe { cbb.dispatch([1, ((n + 31) / 32) as u32, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
