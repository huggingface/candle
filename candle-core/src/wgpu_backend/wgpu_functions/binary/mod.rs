use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, WgpuDevice};

use super::{create_bind_group_input1, create_bind_group_input2, enqueue, MatrixLayout};



#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MetaBinary {
    input1_layout: MatrixLayout,
    input2_layout: MatrixLayout,
    operation: u32,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MetaBinaryContiguousBoth {
    input1_length: u32,
    input1_offset: u32,
    input2_length: u32,
    input2_offset: u32,
    operation: u32,
}

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

pub fn queue_binary_buffer_inplace(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    op: BinaryOperation,
    dtype: crate::DType,
    lay1: crate::Layout,
    lay2: &crate::Layout,
) -> crate::Result<()> {
    let meta = MetaBinary {
        operation: op as u32,
        input1_layout: MatrixLayout::from_layout(&lay1),
        input2_layout: MatrixLayout::from_layout(&lay2),
    };

    let pipeline = dev.get_pipeline(super::Shader::Binary(dtype), Pipelines::BinaryBufferInplace)?;
        
    let bind_group = create_bind_group_input1(dev, pipeline.clone(), meta, buffer_dest, buffer_input1);
    enqueue(
        dev,
        pipeline,
        bind_group,
        lay1.shape().elem_count() as u32,
        #[cfg(feature = "wgpu_debug")] &format!("binary inplace op:{:?}, dtype:{:?}", op, dtype),
    );
    return Ok(());
}

pub fn queue_binary_buffer_from_buffer(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    buffer_input2: &Buffer,
    op: BinaryOperation,
    dtype: crate::DType,
    lay1: &crate::Layout,
    lay2: &crate::Layout,
) -> crate::Result<()> {
    if lay1.is_contiguous() && lay2.is_contiguous() {
        let meta = MetaBinaryContiguousBoth {
            operation: op as u32,
            input1_length: lay1.shape().elem_count() as u32,
            input1_offset: lay1.start_offset()  as u32,
            input2_length: lay2.shape().elem_count()  as u32,
            input2_offset: lay2.start_offset()  as u32,
        };
        let pipeline_type = Pipelines::BinaryBufferFromBufferContiguousBoth;
        let pipeline = dev.get_pipeline(super::Shader::Binary(dtype), pipeline_type.clone())?;

        let bind_group = create_bind_group_input2(
            dev,
            pipeline.clone(),
            meta,
            buffer_dest,
            buffer_input1,
            buffer_input2,
        );
        enqueue(
            dev,
            pipeline,
            bind_group,
            lay1.shape().elem_count() as u32,
            #[cfg(feature = "wgpu_debug")] &format!("binary op:{:?}, dtype:{:?}, pipeline:{:?}", op, dtype, pipeline_type),
        );
        return Ok(());
    } else {
        let meta = MetaBinary {
            operation: op as u32,
            input1_layout: MatrixLayout::from_layout(&lay1),
            input2_layout: MatrixLayout::from_layout(&lay2),
        };

        let pipeline_type = Pipelines::BinaryBufferFromBuffer;
        let pipeline = dev.get_pipeline(super::Shader::Binary(dtype), pipeline_type.clone())?;
        
        let bind_group = create_bind_group_input2(
            dev,
            pipeline.clone(),
            meta,
            buffer_dest,
            buffer_input1,
            buffer_input2,
        );
        enqueue(
            dev,
            pipeline,
            bind_group,
            lay1.shape().elem_count() as u32,
            #[cfg(feature = "wgpu_debug")] &format!("binary op:{:?}, dtype:{:?}, pipeline:{:?}", op, dtype, pipeline_type),
        );
        return Ok(());
    }
}
