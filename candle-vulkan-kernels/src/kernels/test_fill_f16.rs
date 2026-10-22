//! Test kernel: fills an f16 storage buffer with all 65,536 possible
//! 16-bit bit patterns. Thread `i` writes `bit_cast<half>(i as u16)` to
//! element `i`. Used to validate f16 VRAM round-trips end to end.

use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

/// Threads per workgroup in `test_fill_f16.slang` (its `numthreads` is 64).
const WORKGROUP_SIZE: usize = 64;

/// Fills the first `min(total, 65536)` elements of `out` with their index
/// reinterpreted as an f16 bit pattern. `out` must be an f16 storage buffer.
pub fn call_test_fill_f16<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::TestFillF16, KernelName::TestFillF16)?;
    let out_info = DescriptorBufferInfo {
        buffer: Some(out.buffer()),
        offset: out.offset(),
        range: Some(out.size()),
        ..Default::default()
    };
    let set = kernels.descriptor_set(&entry.set_layout, &[out_info])?;

    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups =
        ((total as u64 + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64)) as u32;
    unsafe { cbb.dispatch([workgroups, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
