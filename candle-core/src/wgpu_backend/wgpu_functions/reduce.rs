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

pub struct ReduceParams {
    pub dest_size: u32,
    pub output_to_start_shape_stride2: u32,
    pub output_to_start_stride1: u32,
    pub output_to_start_stride2: u32,
    pub reduction_length: u32,
    pub stride_reduction: u32,
}

pub fn queue_reduce_from_buffer_op(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    op: ReduceOperations,
    dtype: crate::DType,
    layout_input1: &Layout,
    params: ReduceParams,
) -> crate::Result<()> {
    let ReduceParams {
        dest_size,
        output_to_start_shape_stride2,
        output_to_start_stride1,
        output_to_start_stride2,
        reduction_length,
        stride_reduction,
    } = params;

    let mut queue = dev.get_queue();

    let const_vec = vec![op as u32, stride_reduction];

    queue.add(reduction_length);
    queue.add(output_to_start_stride1);
    queue.add(output_to_start_shape_stride2);
    queue.add(output_to_start_stride2);
    queue.add(dest_size);
    queue.add_layout1(layout_input1);

    let use_small_reduce = reduction_length < 16 || stride_reduction != 1;

    if (!use_small_reduce && dest_size > 65535) || (use_small_reduce && dest_size > 65535 * 64) {
        queue.add_const(candle_wgpu_kernels::Constants::UseZ, true);
    }

    let pipeline_type = if use_small_reduce {
        match op {
            ReduceOperations::Sum | ReduceOperations::Min | ReduceOperations::Max => {
                Functions::ReduceSmall
            }
            ReduceOperations::ArgMin | ReduceOperations::ArgMax => Functions::ReduceIndexSmall,
        }
    } else {
        match op {
            ReduceOperations::Sum | ReduceOperations::Min | ReduceOperations::Max => {
                Functions::Reduce
            }
            ReduceOperations::ArgMin | ReduceOperations::ArgMax => Functions::ReduceIndex,
        }
    };

    let pipeline = queue.get_pipeline_const(
        Pipelines::Reduce(dev.get_dtype(dtype)?, pipeline_type),
        const_vec,
    );

    let bind_group = dev.create_bind_group_input1(buffer_dest, buffer_input, dtype.into());

    let y;
    let z;
    if use_small_reduce {
        let dest_size = (dest_size + 63) / 64;
        y = dest_size.min(65535);
        z = (dest_size + 65534) / 65535;
    } else {
        y = dest_size.min(65535);
        z = (dest_size + 65534) / 65535;
    }

    queue.enqueue_workgroups_extra(
        pipeline,
        bind_group,
        1,
        y,
        z,
        (reduction_length * dest_size) as usize,
        #[cfg(feature = "wgpu_debug")]
        Some(format!(
            "layout: {:?} reduction :{}, dest_size: {}",
            layout_input1, reduction_length, dest_size
        )),
    );
    Ok(())
}
