use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, WgpuDevice};

use super::{create_bind_group_input1, enqueue, enqueue_workgroups, flush_gpu_command, MatrixLayout};


#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MetaCopy2d {
    d1: u32,
    d2: u32,
    input1_stride1: u32,
    dest_stride1: u32,
    input1_offset: u32,
    dest_offset: u32,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MetaCopyStrided {
    input1_layout: MatrixLayout,
    dest_offset: u32,
}


pub fn queue_copy_strided(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input: &Buffer,
    dtype: crate::DType,
    input_layout: &crate::Layout,
    dst_offset: u32,
) -> crate::Result<()> {
    if input_layout.shape().elem_count() > 0{
        let meta = MetaCopyStrided {
            input1_layout: MatrixLayout::from_layout(&input_layout),
            dest_offset: dst_offset,
        };
    
        let pipeline = dev.get_pipeline(super::Shader::Copy(dtype), Pipelines::CopyStrided)?;
    
        let bind_group = create_bind_group_input1(dev, pipeline.clone(), meta, buffer_dest, buffer_input);
        enqueue(
            dev,
            pipeline,
            bind_group,
            input_layout.shape().elem_count() as u32,
            &format!("copy strided dtype:{:?}", dtype),
        );
    }
    return Ok(());
}



pub fn queue_copy(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input: &Buffer,
    destination_offset: usize,
    source_offset: usize,
    copy_size: usize,
) {
    if copy_size > 0{
        flush_gpu_command(dev); 
        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        //insert_debug_info_start(dev, &mut encoder);
        encoder.copy_buffer_to_buffer(
            buffer_input,
            source_offset as u64 * 4,
            buffer_dest,
            destination_offset as u64 * 4,
            copy_size as u64 * 4,
        );
        //insert_debug_info_end(dev, &mut encoder, &format!("copy"));
        dev.queue.submit(Some(encoder.finish()));
    }
}




pub fn queue_copy2d(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input: &Buffer,
    dtype: crate::DType,
    d1: u32,
    d2: u32,
    input_stride1: u32,
    dest_stride1: u32,
    input_offset: u32,
    dest_offset: u32,
) -> crate::Result<()> {
    if buffer_dest.size() > 0 && buffer_input.size() > 0{
        let meta = MetaCopy2d {
            d1,
            d2,
            input1_stride1: input_stride1,
            dest_stride1,
            input1_offset: input_offset,
            dest_offset,
        };
    
        let pipeline = dev.get_pipeline(super::Shader::Copy(dtype), Pipelines::Copy2d)?;
    
        let bind_group = create_bind_group_input1(dev, pipeline.clone(), meta, buffer_dest, buffer_input);
        enqueue_workgroups(
            dev,
            pipeline,
            bind_group,
            (d1 + 7) / 8,
            (d2 + 7) / 8,
            1,
            &format!("copy2d dtype:{:?}", dtype),
        );
    }
    return Ok(());
}