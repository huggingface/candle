//! Fused quantized GEMM dispatches:
//! `out[row, col] = lhs[row, k] @ dequant(W[col, k])`.
//!
//! The weight stays quantized in VRAM (no f32 materialization); the kernel
//! dequantizes on the fly. `source`/`name` select the quantization format
//! and the lhs row count: the decode variants (`*Gemm`, `m == 1`) use the
//! single-row kernels, the prefill variants (`*GemmM`, `m > 1`) the
//! multi-row kernels.
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

const TILE: usize = 32;

fn buf_info<T>(buf: &Subbuffer<[T]>) -> DescriptorBufferInfo<'_> {
    DescriptorBufferInfo {
        buffer: Some(buf.buffer()),
        offset: buf.offset(),
        range: Some(buf.size()),
        ..Default::default()
    }
}

/// Records a fused dequant+GEMM dispatch onto `cbb` for an `m`-row lhs.
/// `w` holds the raw quantized bytes of the `(n, k)` weight, `lhs` is the
/// `(m, k)` f32 matrix, `output` is the `(m, n)` f32 result, and `params`
/// holds `[m, n, k]`. Each 32-thread workgroup handles 32 output columns
/// (one thread per column for decode, one thread per output element for
/// prefill; in-register dequantization, no shared memory); `k` must be a
/// multiple of the quantized block size (32 for Q4_0/Q5_0/Q5_1/Q8_0/Q8_1,
/// 256 for Q4_K/Q5_K/Q8_K).
pub fn call_fused_quant_gemm_slang(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    lhs: &Subbuffer<[f32]>,
    w: &Subbuffer<[u8]>,
    output: &Subbuffer<[f32]>,
    params: &Subbuffer<[u64]>,
    m: usize,
    n: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let info0 = buf_info(lhs);
    let info1 = buf_info(output);
    let info2 = buf_info(w);
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
    unsafe { cbb.dispatch([1, (n.div_ceil(TILE)) as u32, m as u32]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
