use crate::wgpuError;

use super::*;

pub fn queue_where_cond(
    dev: &WgpuDevice,
    dest_buffer: BufferReferenceId,
    input_buffer: BufferReferenceId,
    true_buffer: BufferReferenceId,
    false_buffer: BufferReferenceId,
    layout_input: &crate::Layout,
    layout_true: &crate::Layout,
    layout_false: &crate::Layout,
    cond_type: crate::DType,
    dtype: crate::DType,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);
    meta.add_layout1(&layout_input);
    meta.add_layout2(&layout_true);
    meta.add_layout3(&layout_false);

    let (pipeline, cond_alignment) = match cond_type {
        crate::DType::U32 => (Pipelines::WhereCond(get_dtype(dtype)?, candle_wgpu_kernels::where_cond::Functions::WhereCondIndexU32), cond_type.into()),
        crate::DType::I64 => (Pipelines::WhereCondi64(get_dtype(dtype)?, candle_wgpu_kernels::where_condi64::Functions::WhereCondIndexI64), cond_type.into()),
        crate::DType::U8 => (Pipelines::WhereCond(get_dtype(dtype)?, candle_wgpu_kernels::where_cond::Functions::WhereCondIndexU8), crate::DType::U32.into()),
        _ => wgpuError!(format!("dtype: {:?} is not supported for condition in where_cond", cond_type)),
    };
    let pipeline = meta.get_pipeline(pipeline);
    
    let bind_group = create_bind_group_input3_with_alignment(
        dest_buffer,
        input_buffer,
        true_buffer,
        false_buffer,
        BindgroupAlignmentLayout::Bindgroup3(
            dtype.into(),
            cond_alignment,
            dtype.into(),
            dtype.into(),
        ),
    );
    enqueue(
        meta,
        pipeline,
        bind_group,
        layout_input.shape().elem_count() as u32,
        layout_input.shape().elem_count(),
    );
    return Ok(());
}
