use candle_wgpu_kernels::softmax::Functions;

use super::*;

pub fn queue_softmax(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    dtype: crate::DType,
    input1_offset: u32,
    reduction_length: u32,
    dest_size: u32,
) -> crate::Result<()> {
    let const_vec = vec![input1_offset];

    let mut queue = dev.get_queue();
    queue.add(reduction_length);
    queue.add(dest_size);

    let id: u32 = dest_size;
    if id > 65535 {
        queue.add_const(candle_wgpu_kernels::Constants::UseZ, true);
    }

    let pipeline = queue.get_pipeline_const(
        Pipelines::Softmax(dev.get_dtype(dtype)?, Functions::Softmax),
        const_vec,
    );

    let bind_group = dev.create_bind_group_input1(buffer_dest, buffer_input1, dtype.into());

    queue.enqueue_workgroups_extra(
        pipeline,
        bind_group,
        1,
        (id).min(65535),
        id.div_ceil(65535),
        (reduction_length * dest_size) as usize,
        #[cfg(feature = "wgpu_debug")]
        Some(format!("{reduction_length}x{dest_size}({input1_offset})")),
    );

    Ok(())
}
