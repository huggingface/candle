//! Test kernel: validates slang `printf()` debug statements for every data
//! type supported by the backend. See `shaders/test/test_printf.slang` for
//! the buffer layout.
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

/// Runs the `test_printf` kernel: copies the 18 64-bit words from `in_` to
/// `out_` and prints one "Expected ..., actual ..." line per word via
/// slang `printf`. `in_` and `out_` must be 64-bit storage buffers.
pub fn call_test_printf<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    in_: &Subbuffer<[T]>,
    out_: &Subbuffer<[T]>,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::TestPrintf, KernelName::TestPrintf)?;
    let in_info = DescriptorBufferInfo {
        buffer: Some(in_.buffer()),
        offset: in_.offset(),
        range: Some(in_.size()),
        ..Default::default()
    };
    let out_info = DescriptorBufferInfo {
        buffer: Some(out_.buffer()),
        offset: out_.offset(),
        range: Some(out_.size()),
        ..Default::default()
    };
    let writes = [
        WriteDescriptorSet::buffer(0, &in_info),
        WriteDescriptorSet::buffer(1, &out_info),
    ];
    let set = DescriptorSet::new(
        kernels.dss_alloc(),
        &entry.set_layout,
        &writes,
        &[],
    )
    .map_err(|e| match e {
        vulkano::Validated::Error(e) => VulkanKernelError::DescriptorSet(e.to_string()),
        vulkano::Validated::ValidationError(e) => VulkanKernelError::DescriptorSet(format!(
            "validation error: {} (requires: {}; vuids: {:?})",
            e.problem, e.requires_one_of, e.vuids
        )),
    })?;

    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    unsafe { cbb.dispatch([1, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
