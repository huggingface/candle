//! fills a u32 storage buffer with 256 distinct values, each beyond the 16-bit unsigned range. Used to validate u32 VRAM round-trips.
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::kernels::dispatch::dispatch_single_buffer;
use crate::source::Source;
/// Threads per workgroup in `fill_u32.slang` (its `numthreads` is 256).
const WORKGROUP_SIZE: usize = 256;
/// Fills the first `total` elements of `out` with distinct u32 values. `out` must be a u32 storage buffer.
pub fn call_test_fill_u32<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    dispatch_single_buffer(
        cbb,
        kernels,
        Source::TestFillU32,
        KernelName::TestFillU32,
        WORKGROUP_SIZE,
        out,
        total,
    )
}
