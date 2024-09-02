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
    let workgroup_count = u32::min(64, reduction_length);
    let const_vec = vec![workgroup_count, input1_offset];

    let mut meta = get_meta(&dev);
    meta.add(reduction_length);
    meta.add(dest_size);

    let id: u32 = dest_size;
    if id > 65535 {
        meta.add_const(candle_wgpu_kernels::Constants::UseZ, true);
    }

    let pipeline = meta.get_pipeline_const(
        Pipelines::Softmax(get_dtype(dtype)?, Functions::Softmax),
        const_vec,
    );

    let bind_group = create_bind_group_input1(buffer_dest, buffer_input1, dtype.into());

    enqueue_workgroups_extra(
        meta,
        pipeline,
        bind_group,
        1,
        (id).min(65535),
        (id + 65534) / 65535,
        (reduction_length * dest_size) as usize,
        #[cfg(feature = "wgpu_debug")]
        Some(format!("{reduction_length}x{dest_size}({input1_offset})")),
    );

    return Ok(());
}
