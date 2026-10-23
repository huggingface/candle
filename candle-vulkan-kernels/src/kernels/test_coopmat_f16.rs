//! Test kernels for cooperative-matrix (Slang `linalg.CoopMat`) f16
//! support: each probe exercises one part of the path in isolation
//! (fill+Store, Load+Store round-trip, multiply-add) on a single 16x16
//! tile with one 64-thread workgroup.
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

fn buf_info<T>(buf: &Subbuffer<[T]>) -> DescriptorBufferInfo<'_> {
    DescriptorBufferInfo {
        buffer: Some(buf.buffer()),
        offset: buf.offset(),
        range: Some(buf.size()),
    }
}

fn dispatch_1x1(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    source: Source,
    name: KernelName,
    buffers: Vec<DescriptorBufferInfo<'_>>,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(source, name)?;
    let writes: Vec<WriteDescriptorSet<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(i, info)| WriteDescriptorSet::buffer(i as u32, info))
        .collect();
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    unsafe { cbb.dispatch([1, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}

/// Fill+Store probe: writes a 16x16 f16 tile of 1.0 values to `out`
/// through a cooperative-matrix accumulator (`fill` + `Store`).
pub fn call_test_coopmat_fill_store_f16<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    out: &Subbuffer<[T]>,
) -> Result<(), VulkanKernelError> {
    dispatch_1x1(
        cbb,
        kernels,
        Source::TestCoopMatFillStoreF16,
        KernelName::TestCoopMatFillStoreF16,
        vec![buf_info(out)],
    )
}

/// Load+Store probe: copies a 16x16 f16 tile from `input` to `out`
/// through cooperative-matrix `Load`/`Store` (row-major, row stride 16).
pub fn call_test_coopmat_load_store_f16<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    input: &Subbuffer<[T]>,
    out: &Subbuffer<[T]>,
) -> Result<(), VulkanKernelError> {
    dispatch_1x1(
        cbb,
        kernels,
        Source::TestCoopMatLoadStoreF16,
        KernelName::TestCoopMatLoadStoreF16,
        vec![buf_info(input), buf_info(out)],
    )
}

/// Multiply-add probe: C = A @ B (16x16x16, f16 in / f16 accumulate)
/// through a single cooperative-matrix `coopMatMulAdd`.
pub fn call_test_coopmat_muladd_f16<T: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    a: &Subbuffer<[T]>,
    b: &Subbuffer<[T]>,
    c: &Subbuffer<[T]>,
) -> Result<(), VulkanKernelError> {
    dispatch_1x1(
        cbb,
        kernels,
        Source::TestCoopMatMulAddF16,
        KernelName::TestCoopMatMulAddF16,
        vec![buf_info(a), buf_info(b), buf_info(c)],
    )
}

/// Params probe: stores the u64 params values read from `params` into
/// the f16 `out` buffer (out[i] = i < 16 ? params[1] : params[3]).
pub fn call_test_coopmat_params_f16<T: BufferContents, U: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    params: &Subbuffer<[U]>,
    out: &Subbuffer<[T]>,
) -> Result<(), VulkanKernelError> {
    dispatch_1x1(
        cbb,
        kernels,
        Source::TestCoopMatParamsF16,
        KernelName::TestCoopMatParamsF16,
        vec![buf_info(params), buf_info(out)],
    )
}

/// Runtime base/stride probe: Load/Store round-trip with the base and
/// stride read from the u64 `params` buffer (non-constant values).
pub fn call_test_coopmat_rtbase_f16<T: BufferContents, U: BufferContents>(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    input: &Subbuffer<[T]>,
    out: &Subbuffer<[T]>,
    params: &Subbuffer<[U]>,
) -> Result<(), VulkanKernelError> {
    dispatch_1x1(
        cbb,
        kernels,
        Source::TestCoopMatRtBaseF16,
        KernelName::TestCoopMatRtBaseF16,
        vec![buf_info(input), buf_info(out), buf_info(params)],
    )
}
