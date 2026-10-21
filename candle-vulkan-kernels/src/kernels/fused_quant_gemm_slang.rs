//! Fused quantized GEMM dispatches: `out[m, n] = lhs[m, k] @ dequant(W[n, k])`.
//!
//! The weight stays quantized in VRAM (no f32 materialization); the kernel
//! dequantizes on the fly. `source`/`name` select the quantization format
//! (Q4_0 or Q4_K).
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

const TILE: usize = 16;

fn buf_info<T>(buf: &Subbuffer<[T]>) -> DescriptorBufferInfo<'_> {
    DescriptorBufferInfo {
        buffer: Some(buf.buffer()),
        offset: buf.offset(),
        range: Some(buf.size()),
        ..Default::default()
    }
}

/// Records a fused dequant+GEMM dispatch onto `cbb`. `w` holds the raw
/// quantized bytes of the `(n, k)` weight, `output` is `(m, n)` f32,
/// `params` holds `[m, n, k]`. Dispatch grid is `(ceil(m/16), ceil(n/16),
/// 1)` workgroups of 16x16 threads; `k` must be a multiple of the quantized
/// block size (32 for Q4_0, 256 for Q4_K).
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
    unsafe {
        cbb.dispatch([(m.div_ceil(TILE)) as u32, (n.div_ceil(TILE)) as u32, 1])
    }
    .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
