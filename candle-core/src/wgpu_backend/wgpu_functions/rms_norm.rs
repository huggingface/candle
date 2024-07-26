use candle_wgpu_kernels::rms_norm::Functions;

use super::*;

pub fn queue_rms_norm(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_alpha: BufferReferenceId,
    dtype: crate::DType,
    input1_offset : u32,
    alpha_offset : u32,
    reduction_length : u32,
    dest_size: u32,
    eps : f32
) -> crate::Result<()> {
    let workgroup_count = u32::min(64, (reduction_length / 10 + 1) as u32);
    let workgroup_size = reduction_length as u32 / workgroup_count + 1;
    
    let mut meta = get_meta(&dev);

    meta.add(workgroup_count);
    meta.add(workgroup_size);
    meta.add(reduction_length);
    meta.add(input1_offset);
    meta.add(alpha_offset);
    meta.add(eps);

    let pipeline = get_pipeline(Pipelines::RmsNorm(get_dtype(dtype)?, Functions::RmsNorm));

    let bind_group = create_bind_group_input2( buffer_dest, buffer_input1, buffer_alpha);
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        1,
        dest_size,
        1,
        (reduction_length * dest_size) as usize,
    );
    return Ok(());
}
