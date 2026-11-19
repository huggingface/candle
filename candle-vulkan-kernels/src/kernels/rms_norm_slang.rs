//! Slang RMSNorm dispatch: one thread per row (last-dim reduction).
//! Parameters (`n_rows`, `n`, `eps`) travel through the pipeline's push
//! constant block.
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

/// Records a RMSNorm dispatch: `src` is `(n_rows, n)` f32, `alpha` is `(n,)`
/// f32, `dst` is `(n_rows, n)` f32. `params` (`n_rows`, `n`, `eps`) is
/// pushed through the pipeline's push constant block.
pub fn call_rms_norm_slang(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    src: &Subbuffer<[f32]>,
    alpha: &Subbuffer<[f32]>,
    dst: &Subbuffer<[f32]>,
    params: &NormParams,
    n_rows: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let src_info = buf_info(src);
    let alpha_info = buf_info(alpha);
    let dst_info = buf_info(dst);
    let writes = vec![
        WriteDescriptorSet::buffer(0, &src_info),
        WriteDescriptorSet::buffer(1, &alpha_info),
        WriteDescriptorSet::buffer(2, &dst_info),
    ];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
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
