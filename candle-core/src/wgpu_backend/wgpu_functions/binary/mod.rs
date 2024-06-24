use std::sync::Arc;

use crate::{wgpu::{cache::BufferReference, device::{PipelineType, Pipelines}}, WgpuDevice};

use super::{create_bind_group_input2, enqueue, get_meta, get_size};

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

        let (mut meta,  meta_offset) = get_meta(&dev, 5);
        meta.add(op as u32);
        meta.add(lay1.shape().elem_count()); //input1_length
        meta.add(lay1.start_offset()); //input1_offset
        meta.add(lay2.shape().elem_count()); //input2_length
        meta.add(lay2.start_offset()); //input2_offset

        let pipeline_type = Pipelines::BinaryBufferFromBufferContiguousBoth;
        let pipeline = dev.get_pipeline(super::Shader::Binary(dtype), pipeline_type.clone())?;

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
            PipelineType(super::Shader::Binary(dtype),Pipelines::BinaryBufferFromBufferContiguousBoth),
            bind_group,
            lay1.shape().elem_count() as u32,
            #[cfg(feature = "wgpu_debug")] 
            crate::wgpu::device::QueueDebugInfo::new(&format!("binary op:{:?}, dtype:{:?}, pipeline:{:?}", op, dtype, pipeline_type), lay1.shape().elem_count()),
        );
        return Ok(());
    } else {
        let (mut meta,  meta_offset) = get_meta(&dev, 1 + get_size(&lay1) + get_size(&lay2));
        meta.add(op as u32);
        meta.add_layout(&lay1);
        meta.add_layout(&lay2);

        let pipeline_type = Pipelines::BinaryBufferFromBuffer;
        let pipeline = dev.get_pipeline(super::Shader::Binary(dtype), pipeline_type.clone())?;
        
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
            lay1.shape().elem_count() as u32,
            #[cfg(feature = "wgpu_debug")] 
            crate::wgpu::device::QueueDebugInfo::new(&format!("binary op:{:?}, dtype:{:?}, pipeline:{:?}", op, dtype, pipeline_type), lay1.shape().elem_count()),
        );
        return Ok(());
    }
}
