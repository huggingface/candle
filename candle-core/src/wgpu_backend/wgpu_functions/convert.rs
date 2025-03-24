use candle_wgpu_kernels::convert::Functions;

use crate::wgpuError;

use super::*;

pub fn queue_convert_u8_to_f32(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    input_layout: &crate::Layout,
) -> crate::Result<()> {
    let mut queue = dev.get_queue();
    queue.add_layout1(input_layout);

    let pipeline = queue.get_pipeline(Pipelines::Convert(DType::U8, Functions::ConvertU8ToF32));
    let bind_group =
        dev.create_bind_group_input1(buffer_dest, buffer_input, BindgroupAlignment::Aligned4);
    queue.enqueue_64(
        pipeline,
        bind_group,
        input_layout.shape().elem_count() as u32,
        input_layout.shape().elem_count(),
    );
    Ok(())
}

pub fn queue_convert_u32_to_u8(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    start_offset: u32,
    size: u32,
) -> crate::Result<()> {
    let mut queue = dev.get_queue();
    queue.add(start_offset);
    queue.add(size);

    let pipeline = queue.get_pipeline(Pipelines::Convert(DType::U32, Functions::ConvertU32ToU8));

    let bind_group =
        dev.create_bind_group_input1(buffer_dest, buffer_input, BindgroupAlignment::Aligned4);
    queue.enqueue_64(pipeline, bind_group, (size + 3) / 4, size as usize);
    Ok(())
}

pub fn queue_convert_f32_to_u8(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    start_offset: u32,
    size: u32,
) -> crate::Result<()> {
    let mut queue = dev.get_queue();
    queue.add(start_offset);
    queue.add(size);

    let pipeline = queue.get_pipeline(Pipelines::Convert(DType::F32, Functions::ConvertF32ToU8));

    let bind_group =
        dev.create_bind_group_input1(buffer_dest, buffer_input, BindgroupAlignment::Aligned4);
    queue.enqueue_64(pipeline, bind_group, (size + 3) / 4, size as usize);
    Ok(())
}

pub fn queue_convert(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    input_layout: &crate::Layout,
    dest_dtype: crate::DType,
    input_dtype: crate::DType,
) -> crate::Result<()> {
    let mut queue = dev.get_queue();
    queue.add_layout1(input_layout);

    let pipeline = match dest_dtype {
        crate::DType::U32 => Pipelines::Convert(get_dtype(input_dtype)?, Functions::ConvertToU32),
        crate::DType::F32 => Pipelines::Convert(get_dtype(input_dtype)?, Functions::ConvertToF32),
        crate::DType::I64 => Pipelines::ConvertToI64(
            get_dtype(input_dtype)?,
            candle_wgpu_kernels::convert_to_i64::Functions::ConvertToI64,
        ),
        crate::DType::F64 => Pipelines::ConvertToF64(
            get_dtype(input_dtype)?,
            candle_wgpu_kernels::convert_to_f64::Functions::ConvertToF64,
        ),
        _ => wgpuError!(format!("to dtype: {:?} cannot be converted ", dest_dtype)),
    };

    let pipeline = queue.get_pipeline(pipeline);

    let bind_group = dev.create_bind_group_input1_with_alignment(
        buffer_dest,
        buffer_input,
        BindgroupAlignmentLayout::Bindgroup1(dest_dtype.into(), input_dtype.into()),
    );

    queue.enqueue_64(
        pipeline,
        bind_group,
        input_layout.shape().elem_count() as u32,
        input_layout.shape().elem_count(),
    );
    Ok(())
}
