use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, WgpuDevice};

use super::{create_bind_group_input1, enqueue_workgroups, get_meta};

pub fn queue_softmax(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    dtype: crate::DType,
    input1_offset : u32,
    reduction_length : u32,
    dest_size: u32
) -> crate::Result<()> {
    let workgroup_count = u32::min(64, (reduction_length / 10 + 1) as u32);
    let workgroup_size = reduction_length as u32 / workgroup_count + 1;
    
    let (mut meta,  meta_offset) = get_meta(&dev, 4);
    meta.add(workgroup_count);
    meta.add(workgroup_size);
    meta.add(reduction_length);
    meta.add(input1_offset);

    let pipeline = dev.get_pipeline(super::Shader::Softmax(dtype), Pipelines::Softmax)?;

    let bind_group = create_bind_group_input1(dev, pipeline.clone(), meta_offset, buffer_dest, buffer_input1);
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        1,
        dest_size,
        1,
        #[cfg(feature = "wgpu_debug")] 
        crate::wgpu::device::QueueDebugInfo::new(&format!("softmax, dtype:{:?}", dtype), reduction_length * dest_size),
    );
    return Ok(());
}
