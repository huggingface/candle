use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, Layout, WgpuDevice};

use super::{create_bind_group_input1, enqueue_workgroups, MatrixLayout};


#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MetaInfoReduce {
    input_layout: MatrixLayout,
    operation: u32,
    workgroup_count: u32,
    workgroup_size: u32,
    length: u32, //Length of Reduction(e.g count of elements to sum per output),

    output_to_start_stride1: u32, //Stride between each new Output Index

    output_to_start_shape_stride2: u32, //After x Outputs use Stride 2
    output_to_start_stride2: u32,

    stride_reduction: u32, //The Stride to use for elements in Reduction
}


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
    buffer_dest: &Buffer,
    buffer_input: &Buffer,
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
    let meta = MetaInfoReduce {
        operation: op as u32,
        input_layout: MatrixLayout::from_layout(&layout_input1),
        workgroup_count,
        workgroup_size,
        length: reduction_length as u32,
        output_to_start_shape_stride2: output_to_start_shape_stride2,
        output_to_start_stride1: output_to_start_stride1,
        output_to_start_stride2: output_to_start_stride2,
        stride_reduction,
    };
    let pipeline_type = match op {
        ReduceOperations::Sum => Pipelines::Reduce,
        ReduceOperations::Min => Pipelines::Reduce,
        ReduceOperations::Max => Pipelines::Reduce,
        ReduceOperations::ArgMin => Pipelines::ReduceIndex,
        ReduceOperations::ArgMax => Pipelines::ReduceIndex
    };
    let pipeline = dev.get_pipeline(super::Shader::Reduce(dtype), pipeline_type)?;

    let bind_group = create_bind_group_input1(dev, pipeline.clone(), meta, buffer_dest, buffer_input);
    enqueue_workgroups(
        dev,
        pipeline,
        bind_group,
        1,
        dest_size,
        1,
        #[cfg(feature = "wgpu_debug")] &format!("reduce op:{:?}, dtype:{:?}", op, dtype),
    );
    return Ok(());
}
