//! Slang DP4A probe dispatch: verifies the hardware integer-dot-product
//! instruction (`dot4add_u8packed` -> `OpUDot ... PackedVectorFormat4x8Bit`)
//! is reachable, by computing `out[i] = dot4add_u8packed(a[i], b[i], 0)` for
//! a single 256-thread workgroup.
use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

fn buf_info<T>(buf: &Subbuffer<[T]>) -> DescriptorBufferInfo<'_> {
    DescriptorBufferInfo {
        buffer: Some(buf.buffer()),
        offset: buf.offset(),
        range: Some(buf.size()),
        ..Default::default()
    }
}

/// Records the DP4A probe dispatch onto `cbb`. `a` and `b` hold 256 packed
/// u32 (4 u8 each); `out` receives the 256 unsigned u32 dot products.
pub fn call_dp4a_probe_slang(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    a: &Subbuffer<[u32]>,
    b: &Subbuffer<[u32]>,
    out: &Subbuffer<[u32]>,
) -> Result<(), VulkanKernelError> {
    let entry = kernels.load_entry(Source::Dp4aProbe, KernelName::Dp4aProbe)?;
    let a_info = buf_info(a);
    let b_info = buf_info(b);
    let out_info = buf_info(out);
    let writes = vec![
        WriteDescriptorSet::buffer(0, &a_info),
        WriteDescriptorSet::buffer(1, &b_info),
        WriteDescriptorSet::buffer(2, &out_info),
    ];
    let set = DescriptorSet::new(kernels.dss_alloc(), &entry.set_layout, &writes, &[])
        .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;
    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    // No push constants; the shader hardcodes one 256-thread workgroup.
    unsafe { cbb.dispatch([1, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
