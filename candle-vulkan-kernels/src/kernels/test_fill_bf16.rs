//! Test kernels: fill a bfloat16 storage buffer with all 65,536 possible
//! 16-bit bit patterns. `call_test_fill_bf16_native` uses the device's native
//! bfloat16 support; `call_test_fill_bf16_emulated` uses raw 16-bit stores for
//! devices without native bfloat16. Both leave element `i` holding the bit
//! pattern `i`.
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::kernels::dispatch::dispatch_single_buffer;
use crate::source::Source;
/// Threads per workgroup in the bf16 fill shaders (their `numthreads` is 256).
const WORKGROUP_SIZE: usize = 256;
/// Fills the first `min(total, 65536)` elements of `out` with their index
/// reinterpreted as a bfloat16 bit pattern, using native bfloat16.
pub fn call_test_fill_bf16_native<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    dispatch_single_buffer(
        cbb,
        kernels,
        Source::TestFillBf16Native,
        KernelName::TestFillBf16Native,
        WORKGROUP_SIZE,
        out,
        total,
    )
}
/// Fills the first `min(total, 65536)` elements of `out` with their index as a
/// raw 16-bit pattern (no native bfloat16 required).
pub fn call_test_fill_bf16_emulated<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
    total: usize,
) -> Result<(), VulkanKernelError> {
    dispatch_single_buffer(
        cbb,
        kernels,
        Source::TestFillBf16Emulated,
        KernelName::TestFillBf16Emulated,
        WORKGROUP_SIZE,
        out,
        total,
    )
}
