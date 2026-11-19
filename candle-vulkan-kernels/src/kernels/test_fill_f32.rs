//! fills an f32 storage buffer with 256 distinct values, some beyond the f16 range. Used to validate f32 VRAM round-trips.
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::kernels::dispatch::dispatch_single_buffer;
use crate::source::Source;
/// Threads per workgroup in `fill_f32.slang` (its `numthreads` is 256).
const WORKGROUP_SIZE: usize = 256;
/// Fills the first `total` elements of `out` with distinct f32 values. `out` must be a f32 storage buffer.
pub fn call_test_fill_f32<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    dispatch_single_buffer(
        cbb,
        kernels,
        Source::TestFillF32,
        KernelName::TestFillF32,
        WORKGROUP_SIZE,
        out,
        total,
    )
}
