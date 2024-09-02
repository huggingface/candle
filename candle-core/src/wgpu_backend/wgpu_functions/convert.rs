use candle_wgpu_kernels::convert::Functions;

use super::*;

pub fn queue_convert_u32_to_f32(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    input_layout: &crate::Layout,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);
    meta.add_layout1(&input_layout);

    let pipeline = meta.get_pipeline(Pipelines::Convert(DType::U32, Functions::ConvertToF32));
    let bind_group = create_bind_group_input1(buffer_dest, buffer_input, BindgroupAlignment::Aligned4);
    enqueue(
        meta,
        pipeline,
        bind_group,
        input_layout.shape().elem_count() as u32,
        input_layout.shape().elem_count(),
    );
    return Ok(());
}

pub fn queue_convert_u8_to_f32(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    input_layout: &crate::Layout,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);
    meta.add_layout1(&input_layout);

    let pipeline = meta.get_pipeline(Pipelines::Convert(DType::U8, Functions::ConvertU8ToF32));
    let bind_group = create_bind_group_input1(buffer_dest, buffer_input, BindgroupAlignment::Aligned4);
    enqueue(
        meta,
        pipeline,
        bind_group,
        input_layout.shape().elem_count() as u32,
        input_layout.shape().elem_count(),
    );
    return Ok(());
}

pub fn queue_convert_f32_to_u32(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    input_layout: &crate::Layout,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);
    meta.add_layout1(&input_layout);

    let pipeline = meta.get_pipeline(Pipelines::Convert(DType::F32, Functions::ConvertToU32));

    let bind_group = create_bind_group_input1(buffer_dest, buffer_input, BindgroupAlignment::Aligned4);
    enqueue(
        meta,
        pipeline,
        bind_group,
        input_layout.shape().elem_count() as u32,
        input_layout.shape().elem_count(),
    );
    return Ok(());
}

pub fn queue_convert_u32_to_u8(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    start_offset: u32,
    size: u32,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);
    meta.add(start_offset);
    meta.add(size);

    let pipeline = meta.get_pipeline(Pipelines::Convert(DType::U32, Functions::ConvertU32ToU8));

    let bind_group = create_bind_group_input1(buffer_dest, buffer_input, BindgroupAlignment::Aligned4);
    enqueue(
        meta,
        pipeline,
        bind_group,
        ((size + 3) / 4) as u32,
        size as usize,
    );
    return Ok(());
}

pub fn queue_convert_f32_to_u8(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    start_offset: u32,
    size: u32,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);
    meta.add(start_offset);
    meta.add(size);

    let pipeline = meta.get_pipeline(Pipelines::Convert(DType::F32, Functions::ConvertF32ToU8));

    let bind_group = create_bind_group_input1(buffer_dest, buffer_input, BindgroupAlignment::Aligned4);
    enqueue(
        meta,
        pipeline,
        bind_group,
        ((size + 3) / 4) as u32,
        size as usize,
    );
    return Ok(());
}


pub fn queue_convert(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    input_layout: &crate::Layout,
    dest_dtype : crate::DType,
    input_dtype : crate::DType,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);
    meta.add_layout1(&input_layout);

    let function = match dest_dtype{
        crate::DType::U8 => todo!(),
        crate::DType::U32 => Functions::ConvertToU32,
        crate::DType::I64 => Functions::ConvertToI64,
        crate::DType::BF16 => todo!(),
        crate::DType::F16 => todo!(),
        crate::DType::F32 => Functions::ConvertToF32,
        crate::DType::F64 => Functions::ConvertToF64,
    };
    let pipeline = meta.get_pipeline(Pipelines::Convert(get_dtype(input_dtype)?,  function));

    let bind_group = create_bind_group_input1_alignment(buffer_dest, buffer_input, dest_dtype.into(), input_dtype.into());

    enqueue(
        meta,
        pipeline,
        bind_group,
        input_layout.shape().elem_count() as u32,
        input_layout.shape().elem_count(),
    );
    return Ok(());
}
