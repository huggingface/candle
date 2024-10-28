use crate::wgpuError;

use super::*;

pub fn queue_where_cond(
    dev: &WgpuDevice,
    dest_buffer: BufferReferenceId,
    input: WgpuTensor,
    tensor_true: WgpuTensor,
    tensor_false: WgpuTensor,
    cond_type: crate::DType,
    dtype: crate::DType,
) -> crate::Result<()> {
    let mut meta = get_meta(dev);
    meta.add_layout1(input.layout());
    meta.add_layout2(tensor_true.layout());
    meta.add_layout3(tensor_false.layout());

    let (pipeline, cond_alignment) = match cond_type {
        crate::DType::U32 => (
            Pipelines::WhereCond(
                get_dtype(dtype)?,
                candle_wgpu_kernels::where_cond::Functions::WhereCondIndexU32,
            ),
            cond_type.into(),
        ),
        crate::DType::I64 => (
            Pipelines::WhereCondi64(
                get_dtype(dtype)?,
                candle_wgpu_kernels::where_condi64::Functions::WhereCondIndexI64,
            ),
            cond_type.into(),
        ),
        crate::DType::U8 => (
            Pipelines::WhereCond(
                get_dtype(dtype)?,
                candle_wgpu_kernels::where_cond::Functions::WhereCondIndexU8,
            ),
            crate::DType::U32.into(),
        ),
        _ => wgpuError!(format!(
            "dtype: {:?} is not supported for condition in where_cond",
            cond_type
        )),
    };
    let pipeline = meta.get_pipeline(pipeline);

    let bind_group = create_bind_group_input3_with_alignment(
        dest_buffer,
        input.buffer(),
        tensor_true.buffer(),
        tensor_false.buffer(),
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
        input.layout().shape().elem_count() as u32,
        input.layout().shape().elem_count(),
    );
    Ok(())
}
