//! General strided f32 copy: both source and destination may carry
//! arbitrary strides (up to 4 dims, right-aligned).

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;

use crate::err::VulkanKernelError;
use crate::kernel::{KernelName, Kernels};
use crate::source::Source;

const WORKGROUP_SIZE: usize = 256;

/// Copies `total` elements between two strided views of the same shape.
///
/// `dims`, `src_strides` and `dst_strides` are right-aligned over up to 4
/// dimensions (shorter vectors are left-padded with zeros). `src_offset`
/// and `dst_offset` are element offsets into the respective buffers.
pub fn call_copy_f32(
    cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    kernels: &Kernels,
    src: &Subbuffer<[f32]>,
    dst: &Subbuffer<[f32]>,
    total: usize,
    dims: &[usize],
    src_strides: &[usize],
    dst_strides: &[usize],
    src_offset: usize,
    dst_offset: usize,
) -> Result<(), VulkanKernelError> {
    if dims.len() > 4 {
        return Err(VulkanKernelError::CommandBuffer(
            "copy: more than 4 dimensions".into(),
        ));
    }
    let pack4 = |v: &[usize]| -> [u32; 4] {
        let mut out = [0u32; 4];
        for (slot, &d) in out.iter_mut().zip(v) {
            *slot = d as u32;
        }
        out
    };
    let d = pack4(dims);
    let sd = pack4(src_strides);
    let dd = pack4(dst_strides);

    let entry = kernels.load_entry(Source::Copy, KernelName::CopyF32)?;
        let dst_info = DescriptorBufferInfo {
        buffer: Some(dst.buffer()),
        offset: dst.offset(),
        range: Some(dst.size()),
        ..Default::default()
    };
    let src_info = DescriptorBufferInfo {
        buffer: Some(src.buffer()),
        offset: src.offset(),
        range: Some(src.size()),
        ..Default::default()
    };
    let writes = vec![
        WriteDescriptorSet::buffer(0, &src_info),
        WriteDescriptorSet::buffer(1, &dst_info),
    ];
    let set = DescriptorSet::new(
        kernels.dss_alloc(),
        &entry.set_layout,
        &writes,
        &[],
    )
    .map_err(|e| VulkanKernelError::DescriptorSet(e.to_string()))?;

    cbb.bind_pipeline_compute(entry.pipeline.clone())
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.bind_descriptor_sets(PipelineBindPoint::Compute, entry.layout.clone(), 0, set)
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    cbb.push_constants(
        entry.layout.clone(),
        0,
        [
            total as u32,
            dims.len() as u32,
            d[0],
            d[1],
            d[2],
            d[3],
            sd[0],
            sd[1],
            sd[2],
            sd[3],
            dd[0],
            dd[1],
            dd[2],
            dd[3],
            src_offset as u32,
            dst_offset as u32,
        ],
    )
    .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    let workgroups =
        ((total as u64 + (WORKGROUP_SIZE as u64) - 1) / (WORKGROUP_SIZE as u64)) as u32;
    unsafe { cbb.dispatch([workgroups, 1, 1]) }
        .map_err(|e| VulkanKernelError::CommandBuffer(e.to_string()))?;
    Ok(())
}
