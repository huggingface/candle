//! fills an i16 storage buffer with all 65,536 possible 16-bit bit patterns. Thread `i` writes the bit pattern of `i`. Used to validate i16 VRAM round-trips.
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::kernels::dispatch::dispatch_single_buffer;
use crate::source::Source;
/// Threads per workgroup in `fill_i16.slang` (its `numthreads` is 256).
const WORKGROUP_SIZE: usize = 256;
/// Fills the first `total` elements of `out` with distinct i16 values. `out` must be a i16 storage buffer.
pub fn call_test_fill_i16<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    dispatch_single_buffer(
        cbb,
        kernels,
        Source::TestFillI16,
        KernelName::TestFillI16,
        WORKGROUP_SIZE,
        out,
        total,
    )
}
