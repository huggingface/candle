use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, WgpuDevice};

use super::{create_bind_group_input1, enqueue, get_meta, get_size};


// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// #[repr(C)]
// struct MetaConvert {
//     input1_layout: MatrixLayout,
// }



pub fn queue_convert_u32_to_f32(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input: &Buffer,
    input_layout: &crate::Layout,
) -> crate::Result<()> {
    let (mut meta,  meta_offset) = get_meta(&dev, get_size(&input_layout));
    meta.add_layout(&input_layout);

    // let meta = MetaConvert {
    //     input1_layout: MatrixLayout::from_layout(&input_layout),
    // };

    let pipeline = dev.get_pipeline(super::Shader::Convert(crate::DType::U32), Pipelines::ConvertU32ToF32)?;
    let bind_group = create_bind_group_input1(dev, pipeline.clone(), meta_offset, buffer_dest, buffer_input);
    enqueue(
        dev,
        pipeline,
        bind_group,
        input_layout.shape().elem_count() as u32,
        #[cfg(feature = "wgpu_debug")] &format!("u32_to_f32"),
    );
    return Ok(());
}


pub fn queue_convert_u8_to_f32(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input: &Buffer,
    input_layout: &crate::Layout,
) -> crate::Result<()> {
    let (mut meta,  meta_offset) = get_meta(&dev, get_size(&input_layout));
    meta.add_layout(&input_layout);

    // let meta = MetaConvert {
    //     input1_layout: MatrixLayout::from_layout(&input_layout),
    // };

    let pipeline = dev.get_pipeline(super::Shader::Convert(crate::DType::U8), Pipelines::ConvertU8ToF32)?;
    let bind_group = create_bind_group_input1(dev, pipeline.clone(), meta_offset, buffer_dest, buffer_input);
    enqueue(
        dev,
        pipeline,
        bind_group,
        input_layout.shape().elem_count() as u32,
        #[cfg(feature = "wgpu_debug")] &format!("u8_to_f32"),
    );
    return Ok(());
}

pub fn queue_convert_f32_to_u32(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input: &Buffer,
    input_layout: &crate::Layout,
) -> crate::Result<()> {
    let (mut meta,  meta_offset) = get_meta(&dev,  get_size(&input_layout));
    meta.add_layout(&input_layout);

    // let meta = MetaConvert {
    //     input1_layout: MatrixLayout::from_layout(&input_layout),
    // };

    let pipeline = dev.get_pipeline(super::Shader::Convert(crate::DType::F32), Pipelines::ConvertF32ToU32)?;

    let bind_group = create_bind_group_input1(dev, pipeline.clone(), meta_offset, buffer_dest, buffer_input);
    enqueue(
        dev,
        pipeline,
        bind_group,
        input_layout.shape().elem_count() as u32,
        #[cfg(feature = "wgpu_debug")] &format!("f32_to_u32"),
    );
    return Ok(());
}