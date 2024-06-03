use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, WgpuDevice};

use super::{create_bind_group_input3, enqueue, get_meta, get_size};

// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// #[repr(C)]
// struct MetaWhereCond{
//     input1_layout : MatrixLayout,
//     input2_layout : MatrixLayout,
//     input3_layout : MatrixLayout,
    
// }

pub fn queue_where_cond_u32(
    dev: &WgpuDevice,
    dest_buffer: &Buffer,
    input_buffer: &Buffer,
    true_buffer : &Buffer,
    false_buffer : &Buffer,
    layout_input : &crate::Layout,
    layout_true : &crate::Layout,
    layout_false :&crate::Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    let (mut meta,  meta_offset) = get_meta(&dev, get_size(&layout_input) + get_size(&layout_true) + get_size(&layout_false));
    meta.add_layout(&layout_input);
    meta.add_layout(&layout_true);
    meta.add_layout(&layout_false);

    // let meta = MetaWhereCond {
    //     input1_layout: MatrixLayout::from_layout(&layout_input),
    //     input2_layout: MatrixLayout::from_layout(&layout_true),
    //     input3_layout: MatrixLayout::from_layout(&layout_false),
    // };
    let pipeline = dev.get_pipeline(super::Shader::WhereCond(dtype), Pipelines::WhereCondU32)?;

    let bind_group = create_bind_group_input3(dev, pipeline.clone(), meta_offset,dest_buffer, input_buffer, true_buffer, false_buffer);
    enqueue(
        dev,
        pipeline,
        bind_group,
        layout_input.shape().elem_count() as u32,
        #[cfg(feature = "wgpu_debug")] &format!("where cond u32, dtype:{:?}", Pipelines::WhereCondU32),
    );
    return Ok(());
}