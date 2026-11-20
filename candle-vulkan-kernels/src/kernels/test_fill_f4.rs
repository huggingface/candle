//! Fills a 4-bit E2M1 (nibble) storage buffer with all 16 possible values. Thread
//! `i` writes byte `i` (for `i` in 0..16). Used to validate F4
//! VRAM round-trips end to end.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::kernels::dispatch::dispatch_single_buffer;
use crate::source::Source;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
/// Threads per workgroup in `fill_f4.slang` (its `numthreads` is 256).
const WORKGROUP_SIZE: usize = 256;
/// Fills the first `total` elements of `out` with distinct f4 values. `out`
/// must be a one-byte-per-element storage buffer (e.g. microfloat::f4).
pub fn call_test_fill_f4<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    dispatch_single_buffer(
        cbb,
        kernels,
        Source::TestFillF4,
        KernelName::TestFillF4,
        WORKGROUP_SIZE,
        out,
        total,
    )
}
