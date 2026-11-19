//! Fills a 6-bit E2M3 storage buffer with all 64 possible values. Thread
//! `i` writes byte `i` (for `i` in 0..64). Used to validate F6E2M3
//! VRAM round-trips end to end.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::kernels::dispatch::dispatch_single_buffer;
use crate::source::Source;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
/// Threads per workgroup in `fill_f6e2m3.slang` (its `numthreads` is 256).
const WORKGROUP_SIZE: usize = 256;
/// Fills the first `total` elements of `out` with distinct f6e2m3 values. `out`
/// must be a one-byte-per-element storage buffer (e.g. microfloat::f6e2m3).
pub fn call_test_fill_f6e2m3<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    dispatch_single_buffer(
        cbb,
        kernels,
        Source::TestFillF6e2m3,
        KernelName::TestFillF6e2m3,
        WORKGROUP_SIZE,
        out,
        total,
    )
}
