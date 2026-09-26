//! Slang where_cond dispatch (u8 pred, dtype values -> dtype output).
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

/// Records a where_cond dispatch onto `cbb`:
/// `output[i] = pred[i] != 0 ? true[t_off(i)] : false[f_off(i)]`. `pred` is
/// `u8`; `on_true`, `on_false` and `output` share the element type `T`.
/// `params` must hold `[total, ndim, pred dims, true dims, false dims]`.
pub fn call_where_slang<T>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    pred: &Subbuffer<[u8]>,
    on_true: &Subbuffer<[T]>,
    on_false: &Subbuffer<[T]>,
    output: &Subbuffer<[T]>,
    params: &Subbuffer<[f32]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let pred_info = DescriptorBufferInfo {
        buffer: Some(pred.buffer()),
        offset: pred.offset(),
        range: Some(pred.size()),
        ..Default::default()
    };
    let true_info = DescriptorBufferInfo {
        buffer: Some(on_true.buffer()),
        offset: on_true.offset(),
        range: Some(on_true.size()),
        ..Default::default()
    };
    let false_info = DescriptorBufferInfo {
        buffer: Some(on_false.buffer()),
        offset: on_false.offset(),
        range: Some(on_false.size()),
        ..Default::default()
    };
    let output_info = DescriptorBufferInfo {
        buffer: Some(output.buffer()),
        offset: output.offset(),
        range: Some(output.size()),
        ..Default::default()
    };
    let params_info = DescriptorBufferInfo {
        buffer: Some(params.buffer()),
        offset: params.offset(),
        range: Some(params.size()),
        ..Default::default()
    };
    let writes = vec![
        WriteDescriptorSet::buffer(0, &pred_info),
        WriteDescriptorSet::buffer(1, &true_info),
        WriteDescriptorSet::buffer(2, &false_info),
        WriteDescriptorSet::buffer(3, &output_info),
        WriteDescriptorSet::buffer(4, &params_info),
    ];
    let set = DescriptorSet::new(
        kernels.dss_alloc(),
        &entry.set_layout,
        &writes,
        &[],
    )
    .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups = (total as u64 + 255) / 256;
    unsafe { cbb.dispatch([workgroups as u32, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
