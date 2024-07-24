use std::sync::Arc;

use crate::{wgpu::{cache::BufferReference, device::Pipelines}, WgpuDevice};

use super::{create_bind_group_input2, enqueue_big, get_meta};

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub enum BinaryOperation {
    SetY = 0,
    Add = 1,
    Mult = 2,
    Minus = 3,
    Div = 4,
    Max = 5,
    Min = 6,
    Pow = 7,
}
pub fn queue_binary_buffer_from_buffer(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input1: Arc<BufferReference>,
    buffer_input2: Arc<BufferReference>,
    op: BinaryOperation,
    dtype: crate::DType,
    lay1: &crate::Layout,
    lay2: &crate::Layout,
) -> crate::Result<()> {
    if lay1.is_contiguous() && lay2.is_contiguous() {
        let const_vec = vec![
            op as usize,
            lay1.start_offset(),
            lay2.start_offset()];

        let mut meta = get_meta(&dev);
        //meta.add(op as u32);
        meta.add(lay1.shape().elem_count()); //input1_length
        //meta.add(lay1.start_offset()); //input1_offset
        //meta.add(lay2.shape().elem_count()); //input2_length
        //meta.add(lay2.start_offset()); //input2_offset

        let bind_group = create_bind_group_input2(
            buffer_dest,
            buffer_input1,
            buffer_input2,
        );
        let pipeline = dev.get_pipeline_const(super::Shader::Binary(dtype), Pipelines::BinaryBufferFromBufferContiguousBoth, const_vec);
        enqueue_big(
            meta,
            pipeline,
            bind_group,
            lay1.shape().elem_count() as u32, 
            #[cfg(feature = "wgpu_debug")] 
            crate::wgpu::device::QueueDebugInfo::new(&format!("binary op:{:?}, dtype:{:?}, pipeline:{:?}", op, dtype, Pipelines::BinaryBufferFromBufferContiguousBoth)),
        );
        return Ok(());
    } else {
        let const_vec = vec![
            op as usize];

        let mut meta = get_meta(&dev);
        meta.add_layout(&lay1);
        meta.add_layout(&lay2);

        let pipeline_type = Pipelines::BinaryBufferFromBuffer;
        let pipeline = dev.get_pipeline_const(super::Shader::Binary(dtype), pipeline_type.clone(), const_vec);
        
        let bind_group = create_bind_group_input2(
            buffer_dest,
            buffer_input1,
            buffer_input2,
        );

        enqueue_big(
            meta,
            pipeline,
            bind_group,
            lay1.shape().elem_count() as u32,
            #[cfg(feature = "wgpu_debug")] 
            crate::wgpu::device::QueueDebugInfo::new(&format!("binary op:{:?}, dtype:{:?}, pipeline:{:?}", op, dtype, pipeline_type)),
        );
        return Ok(());
    }
}
