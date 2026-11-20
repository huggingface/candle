//! Fills a FP8 E8M0 scale (emulated as raw bytes) storage buffer with all 256 possible values. Thread
//! `i` writes byte `i` (for `i` in 0..256). Used to validate F8E8M0
//! VRAM round-trips end to end.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::kernels::dispatch::dispatch_single_buffer;
use crate::source::Source;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
/// Threads per workgroup in `fill_f8e8m0.slang` (its `numthreads` is 256).
const WORKGROUP_SIZE: usize = 256;
/// Fills the first `total` elements of `out` with distinct f8e8m0 values. `out`
/// must be a one-byte-per-element storage buffer (e.g. microfloat::f8e8m0).
pub fn call_test_fill_f8e8m0<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    dispatch_single_buffer(
        cbb,
        kernels,
        Source::TestFillF8e8m0,
        KernelName::TestFillF8e8m0,
        WORKGROUP_SIZE,
        out,
        total,
    )
}
