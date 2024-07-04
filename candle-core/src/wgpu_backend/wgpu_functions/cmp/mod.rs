use std::sync::Arc;


use crate::{wgpu::{cache::BufferReference, device::Pipelines}, Layout, WgpuDevice};

use super::{create_bind_group_input2, enqueue, get_meta};

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
    buffer_dest: Arc<BufferReference>,
    buffer_input1: Arc<BufferReference>,
    buffer_input2: Arc<BufferReference>,
    op: CmpOperation,
    dtype: crate::DType,
    layout_input1: &Layout,
    layout_input2: &Layout,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);
    meta.add(op as u32);
    meta.add_layout(&layout_input1);
    meta.add_layout(&layout_input2);

    let pipeline = dev.get_pipeline(super::Shader::Cmp(dtype), Pipelines::CmpFromBuffer)?;

    let bind_group = create_bind_group_input2(
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
