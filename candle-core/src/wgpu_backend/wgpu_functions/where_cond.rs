use super::*;
use candle_wgpu_kernels::where_cond::Functions;

pub fn queue_where_cond_u32(
    dev: &WgpuDevice,
    dest_buffer: BufferReferenceId,
    input_buffer: BufferReferenceId,
    true_buffer: BufferReferenceId,
    false_buffer: BufferReferenceId,
    layout_input: &crate::Layout,
    layout_true: &crate::Layout,
    layout_false: &crate::Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);
    meta.add_layout1(&layout_input);
    meta.add_layout2(&layout_true);
    meta.add_layout3(&layout_false);

    let pipeline = meta.get_pipeline(Pipelines::WhereCond(
        get_dtype(dtype)?,
        Functions::WhereCondIndexU32,
    ));

    let bind_group = create_bind_group_input3(dest_buffer, input_buffer, true_buffer, false_buffer);
    enqueue(
        meta,
        pipeline,
        bind_group,
        layout_input.shape().elem_count() as u32,
        layout_input.shape().elem_count(),
    );
    return Ok(());
}
