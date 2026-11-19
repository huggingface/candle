//! fills an i64 storage buffer with 256 distinct values, each beyond the 32-bit signed range. Used to validate i64 VRAM round-trips.
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::kernels::dispatch::dispatch_single_buffer;
use crate::source::Source;
/// Threads per workgroup in `fill_i64.slang` (its `numthreads` is 256).
const WORKGROUP_SIZE: usize = 256;
/// Fills the first `total` elements of `out` with distinct i64 values. `out` must be a i64 storage buffer.
pub fn call_test_fill_i64<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    dispatch_single_buffer(
        cbb,
        kernels,
        Source::TestFillI64,
        KernelName::TestFillI64,
        WORKGROUP_SIZE,
        out,
        total,
    )
}
