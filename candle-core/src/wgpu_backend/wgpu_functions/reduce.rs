use candle_wgpu_kernels::reduce::Functions;

use super::*;

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
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
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
    let mut meta = get_meta(&dev);

    let const_vec = vec![op as u32, stride_reduction];

    meta.add(reduction_length);
    meta.add(output_to_start_stride1);
    meta.add(output_to_start_shape_stride2);
    meta.add(output_to_start_stride2);
    meta.add(dest_size);
    meta.add_layout1(layout_input1);

    if dest_size > 65535 {
        meta.add_const(candle_wgpu_kernels::Constants::UseZ, true);
    }

    let pipeline_type = match op {
        ReduceOperations::Sum => Functions::Reduce,
        ReduceOperations::Min => Functions::Reduce,
        ReduceOperations::Max => Functions::Reduce,
        ReduceOperations::ArgMin => Functions::ReduceIndex,
        ReduceOperations::ArgMax => Functions::ReduceIndex,
    };
    let pipeline = meta.get_pipeline_const(
        Pipelines::Reduce(get_dtype(dtype)?, pipeline_type),
        const_vec,
    );

    let bind_group = create_bind_group_input1(buffer_dest, buffer_input, dtype.into());

    let y = dest_size.min(65535);
    let z = (dest_size + 65534) / 65535;

    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        1,
        y,
        z,
        (reduction_length * dest_size) as usize,
    );
    return Ok(());
}
