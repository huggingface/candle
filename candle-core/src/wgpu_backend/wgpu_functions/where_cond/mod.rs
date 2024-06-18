use std::sync::Arc;


use crate::{wgpu::{cache::BufferReference, device::Pipelines}, WgpuDevice};

use super::{create_bind_group_input3, enqueue, get_meta, get_size};

pub fn queue_where_cond_u32(
    dev: &WgpuDevice,
    dest_buffer: Arc<BufferReference>,
    input_buffer: Arc<BufferReference>,
    true_buffer : Arc<BufferReference>,
    false_buffer : Arc<BufferReference>,
    layout_input : &crate::Layout,
    layout_true : &crate::Layout,
    layout_false :&crate::Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    let (mut meta,  meta_offset) = get_meta(&dev, get_size(&layout_input) + get_size(&layout_true) + get_size(&layout_false));
    meta.add_layout(&layout_input);
    meta.add_layout(&layout_true);
    meta.add_layout(&layout_false);

    let pipeline = dev.get_pipeline(super::Shader::WhereCond(dtype), Pipelines::WhereCondU32)?;

    let bind_group = create_bind_group_input3(dev, pipeline.clone(), meta_offset,dest_buffer, input_buffer, true_buffer, false_buffer);
    enqueue(
        meta,
        pipeline,
        bind_group,
        layout_input.shape().elem_count() as u32,
        #[cfg(feature = "wgpu_debug")] 
        crate::wgpu::device::QueueDebugInfo::new(&format!("where cond u32, dtype:{:?}", Pipelines::WhereCondU32), layout_input.shape().elem_count()),
    );
    return Ok(());
}