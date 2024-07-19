use std::sync::Arc;


use crate::{wgpu::{cache::BufferReference, device::Pipelines}, WgpuDevice};

use super::{create_bind_group_input1, enqueue_workgroups, get_meta};

pub fn queue_softmax(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input1: Arc<BufferReference>,
    dtype: crate::DType,
    input1_offset : u32,
    reduction_length : u32,
    dest_size: u32
) -> crate::Result<()> {
    let workgroup_count = u32::min(64, (reduction_length / 10 + 1) as u32);
    let workgroup_size = reduction_length as u32 / workgroup_count + 1;
    let workgroup_count = (reduction_length + (workgroup_size - 1)) / workgroup_size; 
    
    let const_vec = vec![
        workgroup_count,
        input1_offset];

    let mut meta = get_meta(&dev);
    //meta.add(workgroup_count);
    meta.add(workgroup_size);
    meta.add(reduction_length);
    //meta.add(input1_offset);
    meta.add(dest_size);
    
    let pipeline = dev.get_pipeline_const(super::Shader::Softmax(dtype), Pipelines::Softmax, const_vec);

    let bind_group: crate::wgpu_backend::cache::BindGroupReferenceBase<Arc<BufferReference>> = create_bind_group_input1( buffer_dest, buffer_input1);
    let id: u32 = dest_size;
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        1,
        (id).min(65535),
        (id + 65534) / 65535,
        (reduction_length * dest_size) as usize,
        #[cfg(feature = "wgpu_debug")] 
        crate::wgpu::device::QueueDebugInfo::new(&format!("softmax, dtype:{:?}, reduction: {reduction_length}, dest_size: {dest_size}", dtype)),
    );
    return Ok(());
}
