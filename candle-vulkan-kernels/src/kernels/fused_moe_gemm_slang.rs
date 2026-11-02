//! Fused routed-MoE quantized GEMM dispatch:
//! `out[route, n] = lhs[route, k] @ dequant(W[ids[route], n, k])`.
//!
//! The weight stays quantized in VRAM (no f32 materialization); each
//! thread dequantizes its (route, column) weight in registers while
//! accumulating the dot product. `lhs` is the `(num_routes, k)` f32
//! matrix (the per-route activations), `ids` the `(num_routes,)` u32
//! expert ids, `w` the raw quantized bytes of the `(experts, n, k)`
//! weight, and `output` the `(num_routes, n)` f32 result. `params` holds
//! `(n, k, weight_expert_bytes)`. Each 32-thread workgroup handles 32
//! output columns for one route; `k` must be a multiple of the quantized
//! block size (32 for Q8_0, 256 for the K-quants).
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::push_constants::FusedMoeGemmParams;
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

/// Records a fused routed-MoE dequant+GEMM dispatch onto `cbb`. `lhs` is
/// the `(num_routes, k)` f32 matrix, `w` the raw quantized bytes of the
/// `(experts, n, k)` weight, `ids` the `(num_routes,)` u32 expert ids,
/// and `output` the `(num_routes, n)` f32 result. `params` is pushed
/// through the pipeline's push constant block. Each 32-thread workgroup
/// handles 32 output columns for one route; `k` must be a multiple of the
/// quantized block size (32 for Q8_0, 256 for the K-quants).
pub fn call_fused_moe_gemm_slang(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    lhs: &Subbuffer<[f32]>,
    output: &Subbuffer<[f32]>,
    w: &Subbuffer<[u8]>,
    ids: &Subbuffer<[u32]>,
    params: &FusedMoeGemmParams,
    n: usize,
    num_routes: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let info0 = buf_info(lhs);
    let info1 = buf_info(output);
    let info2 = buf_info(w);
    let info3 = buf_info(ids);

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
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    unsafe { cbb.dispatch([1, (n.div_ceil(TILE)) as u32, num_routes as u32]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
