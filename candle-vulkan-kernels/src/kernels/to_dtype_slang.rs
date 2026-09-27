//! Slang `to_dtype` dispatch: `g_Out[i] = convert(g_In[i])` for `i < g_Params[0]`.
//! `input` and `output` may have different element types (the source and
//! destination dtype GPU storage types). `params` must hold `[total as f32]`.
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;
/// Records a Slang `to_dtype` dispatch onto `cbb`: `output[i] = convert(input[i])`
/// for `i < total`. `input` has element type `S` and `output` has element type `D`
/// (the source and destination dtype GPU storage types). `params` holds `[total]`.
pub fn call_to_dtype_slang<S, D>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    input: &Subbuffer<[S]>,
    output: &Subbuffer<[D]>,
    params: &Subbuffer<[f32]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let input_info = DescriptorBufferInfo {
        buffer: Some(input.buffer()),
        offset: input.offset(),
        range: Some(input.size()),
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
        WriteDescriptorSet::buffer(0, &input_info),
        WriteDescriptorSet::buffer(1, &output_info),
        WriteDescriptorSet::buffer(2, &params_info),
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
