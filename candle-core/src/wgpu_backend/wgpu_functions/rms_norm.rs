use candle_wgpu_kernels::rms_norm::Functions;

use super::*;

#[allow(clippy::too_many_arguments)]
pub fn queue_rms_norm(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: (BufferReferenceId, u32),
    buffer_alpha: (BufferReferenceId, u32),
    dtype: crate::DType,
    reduction_length: u32,
    dest_size: u32,
    eps: f32,
) -> crate::Result<()> {
    let (buffer_input1, input1_offset) = buffer_input1;
    let (buffer_alpha, alpha_offset) = buffer_alpha;

    let workgroup_count = u32::min(64, reduction_length / 10 + 1);
    let workgroup_size = reduction_length / workgroup_count + 1;

    let mut meta = get_queue(dev);

    meta.add(workgroup_count);
    meta.add(workgroup_size);
    meta.add(reduction_length);
    meta.add(input1_offset);
    meta.add(alpha_offset);
    meta.add(eps);

    let pipeline = meta.get_pipeline(Pipelines::RmsNorm(get_dtype(dtype)?, Functions::RmsNorm));

    let bind_group =
        create_bind_group_input2(buffer_dest, buffer_input1, buffer_alpha, dtype.into());
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        1,
        dest_size,
        1,
        (reduction_length * dest_size) as usize,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn queue_layer_norm(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: (BufferReferenceId, u32),
    buffer_alpha: (BufferReferenceId, u32),
    buffer_beta: (BufferReferenceId, u32),
    dtype: crate::DType,
    reduction_length: u32,
    dest_size: u32,
    eps: f32,
) -> crate::Result<()> {
    let (buffer_input1, input1_offset) = buffer_input1;
    let (buffer_alpha, alpha_offset) = buffer_alpha;
    let (buffer_beta, beta_offset) = buffer_beta;

    let workgroup_count = u32::min(64, reduction_length / 10 + 1);
    let workgroup_size = reduction_length / workgroup_count + 1;

    let mut meta = get_queue(dev);

    meta.add(workgroup_count);
    meta.add(workgroup_size);
    meta.add(reduction_length);
    meta.add(input1_offset);
    meta.add(alpha_offset);
    meta.add(eps);
    meta.add(beta_offset);

    let pipeline = meta.get_pipeline(Pipelines::RmsNorm(get_dtype(dtype)?, Functions::LayerNorm));

    let bind_group = create_bind_group_input3(
        buffer_dest,
        buffer_input1,
        buffer_alpha,
        buffer_beta,
        dtype.into(),
    );
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        1,
        dest_size,
        1,
        (reduction_length * dest_size) as usize,
    );
    Ok(())
}
