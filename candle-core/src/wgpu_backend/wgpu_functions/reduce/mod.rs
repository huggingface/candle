use std::sync::Arc;


use crate::{wgpu::{cache::BufferReference, device::Pipelines}, Layout, WgpuDevice};

use super::{create_bind_group_input1, enqueue_workgroups, get_meta, get_size};

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub enum ReduceOperations {
    Sum = 0,
    Min = 1,
    Max = 2,
    ArgMin = 3,
    ArgMax = 4,
}


pub fn queue_reduce_from_buffer_op(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input: Arc<BufferReference>,
    op: ReduceOperations,
    dtype: crate::DType,
    layout_input1: &Layout,
    dest_size: u32,
    output_to_start_shape_stride2: u32,
    output_to_start_stride1: u32,
    output_to_start_stride2: u32,
    reduction_length: u32,
    stride_reduction: u32,
) -> crate::Result<()> {
    let workgroup_count = u32::min(64, (reduction_length / 10 + 1) as u32);
    let workgroup_size = reduction_length as u32 / workgroup_count + 1;
    
    let (mut meta,  meta_offset) = get_meta(&dev, 8 + get_size(&layout_input1));

    meta.add(op as u32);
    meta.add(workgroup_count);
    meta.add(workgroup_size);
    meta.add(reduction_length);
    meta.add(output_to_start_stride1);
    meta.add(output_to_start_shape_stride2);
    meta.add(output_to_start_stride2);
    meta.add(stride_reduction);
    meta.add_layout(layout_input1);

    let pipeline_type = match op {
        ReduceOperations::Sum => Pipelines::Reduce,
        ReduceOperations::Min => Pipelines::Reduce,
        ReduceOperations::Max => Pipelines::Reduce,
        ReduceOperations::ArgMin => Pipelines::ReduceIndex,
        ReduceOperations::ArgMax => Pipelines::ReduceIndex
    };
    let pipeline = dev.get_pipeline(super::Shader::Reduce(dtype), pipeline_type)?;

    let bind_group = create_bind_group_input1(dev, pipeline.clone(), meta_offset, buffer_dest, buffer_input);
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        1,
        dest_size,
        1,
        #[cfg(feature = "wgpu_debug")] 
        crate::wgpu::device::QueueDebugInfo::new(&format!("reduce op:{:?}, dtype:{:?}", op, dtype), reduction_length * dest_size),
    );
    return Ok(());
}
