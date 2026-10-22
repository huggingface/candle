//! Slang fused add + RMSNorm dispatch: one thread per row (last-dim
//! reduction). Parameters (`n_rows`, `n`, `eps`) travel through the
//! pipeline's push constant block.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::push_constants::NormParams;
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

/// Records a fused add + RMSNorm dispatch: `x` and `h` are `(n_rows, n)` f32
/// (the residual and the add operand), `alpha` is `(n,)` f32, and the
/// outputs are `residual` (`x + h`) and `norm` (RMSNorm of `x + h` times
/// `alpha`), both `(n_rows, n)` f32. `params` (`n_rows`, `n`, `eps`) is
/// pushed through the pipeline's push constant block.
pub fn call_add_rms_norm_slang(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    x: &Subbuffer<[f32]>,
    h: &Subbuffer<[f32]>,
    alpha: &Subbuffer<[f32]>,
    residual: &Subbuffer<[f32]>,
    norm: &Subbuffer<[f32]>,
    params: &NormParams,
    n_rows: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let x_info = buf_info(x);
    let h_info = buf_info(h);
    let alpha_info = buf_info(alpha);
    let residual_info = buf_info(residual);
    let norm_info = buf_info(norm);
    let set = kernels.descriptor_set(&entry.set_layout, &[x_info, h_info, alpha_info, residual_info, norm_info])?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(entry.layout.clone(), 0, *params)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    unsafe { cbb.dispatch([n_rows as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
