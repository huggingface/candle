use std::sync::Arc;


use crate::{wgpu::{device::Pipelines, BufferReferenceId}, Layout, WgpuDevice};

use super::{create_bind_group_input2, enqueue, get_meta, get_size};

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub enum CmpOperation {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}


pub fn queue_cmp_buffer_from_buffer(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    op: CmpOperation,
    dtype: crate::DType,
    layout_input1: &Layout,
    layout_input2: &Layout,
) -> crate::Result<()> {
    let (mut meta,  meta_offset) = get_meta(&dev, 1 + get_size(&layout_input1) + get_size(&layout_input2));
    meta.add(op as u32);
    meta.add_layout(&layout_input1);
    meta.add_layout(&layout_input2);

    let pipeline = dev.get_pipeline(super::Shader::Cmp(dtype), Pipelines::CmpFromBuffer)?;

    let bind_group = create_bind_group_input2(
        dev,
        pipeline.clone(),
        meta_offset,
        buffer_dest,
        buffer_input1,
        buffer_input2,
    );
    enqueue(
        meta,
        pipeline,
        bind_group,
        ((layout_input1.shape().elem_count() + 3) / 4) as u32,
        #[cfg(feature = "wgpu_debug")] 
        crate::wgpu::device::QueueDebugInfo::new(&format!("cmp op:{:?}, dtype:{:?}", op, dtype), layout_input1.shape().elem_count()),
    );
    return Ok(());
}
