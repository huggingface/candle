use std::sync::Arc;

use crate::{wgpu::{cache::BufferReference, device::Pipelines}, WgpuDevice};

use super::{create_bind_group_input1, enqueue, enqueue_workgroups, get_meta, get_size};

pub fn queue_copy_strided(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input: Arc<BufferReference>,
    dtype: crate::DType,
    input_layout: &crate::Layout,
    dst_offset: u32,
) -> crate::Result<()> {
    if input_layout.shape().elem_count() > 0{
        let (mut meta,  meta_offset) = get_meta(&dev, 1 + get_size(&input_layout));
        meta.add(dst_offset);
        meta.add_layout(&input_layout);
    
        let pipeline = dev.get_pipeline(super::Shader::Copy(dtype), Pipelines::CopyStrided)?;
    
        let bind_group = create_bind_group_input1(meta_offset, buffer_dest, buffer_input);
        enqueue(
            meta,
            pipeline,
            bind_group,
            input_layout.shape().elem_count() as u32,
            #[cfg(feature = "wgpu_debug")] 
            crate::wgpu::device::QueueDebugInfo::new(&format!("copy strided dtype:{:?}", dtype), input_layout.shape().elem_count()),
        );
    }
    return Ok(());
}


//This is ~30% faster than using a shader to copy, but a shader dispatch call can be easier cached. therefore we just use the slower copy function at the moment.
//In addition the copy is often not the bottle neck(but matmul or conv-dispatch call)
// pub fn queue_copy_old(
//     dev: &WgpuDevice,
//     buffer_dest: Arc<BufferReference>,
//     buffer_input: Arc<BufferReference>,
//     destination_offset: usize,
//     source_offset: usize,
//     copy_size: usize,
// ) {
//     if copy_size > 0{
//         flush_gpu_command(dev, &mut dev.meta_array.lock().unwrap());

//         #[cfg(feature = "wgpu_debug")]
//         let (global_index, query_set) = super::init_debug_queue(dev,  2);
        

//         let mut encoder = dev
//             .device
//             .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
//         #[cfg(feature = "wgpu_debug")]
//         encoder.write_timestamp(&query_set, 0);
//         encoder.copy_buffer_to_buffer(
//             buffer_input,
//             source_offset as u64 * 4,
//             buffer_dest,
//             destination_offset as u64 * 4,
//             copy_size as u64 * 4,
//         );
//         #[cfg(feature = "wgpu_debug")]
//         encoder.write_timestamp(&query_set, 1);
//         #[cfg(feature = "wgpu_debug")]
//         dev.debug.insert_info(global_index,("copy".to_owned(), copy_size as u64, 0, 0, 0));
//         #[cfg(feature = "wgpu_debug")]
//         super::end_debug_queue(dev, 2, global_index, &mut encoder, &query_set);
//         dev.queue.submit(Some(encoder.finish()));
//     }
// }




pub fn queue_copy(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input: Arc<BufferReference>,
    destination_offset: usize,
    source_offset: usize,
    copy_size: usize,
    dtype : crate::DType
) -> crate::Result<()> {
    if copy_size > 0{
        let (mut meta,  meta_offset) = get_meta(&dev, 3);
        meta.add(copy_size);
        meta.add(destination_offset);
        meta.add(source_offset);
        
        let pipeline = dev.get_pipeline(super::Shader::Copy(dtype), Pipelines::Copy)?;
    
        let bind_group = create_bind_group_input1(meta_offset, buffer_dest, buffer_input);
        enqueue(
            meta,
            pipeline,
            bind_group,
            copy_size as u32,
            #[cfg(feature = "wgpu_debug")] 
            crate::wgpu::device::QueueDebugInfo::new(&format!("copy strided dtype:{:?}", dtype), input_layout.shape().elem_count()),
        );
    }
    return Ok(());
}




pub fn queue_copy2d(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input: Arc<BufferReference>,
    dtype: crate::DType,
    d1: u32,
    d2: u32,
    input_stride1: u32,
    dest_stride1: u32,
    input_offset: u32,
    dest_offset: u32,
) -> crate::Result<()> {
    if buffer_dest.size > 0 && buffer_input.size > 0{
        
        let (mut meta,  meta_offset) = get_meta(&dev, 6);
        meta.add(d1);
        meta.add(d2);
        meta.add(input_stride1);
        meta.add(dest_stride1);
        meta.add(input_offset);
        meta.add(dest_offset);

        let pipeline = dev.get_pipeline(super::Shader::Copy(dtype), Pipelines::Copy2d)?;
    
        let bind_group = create_bind_group_input1(meta_offset, buffer_dest, buffer_input);
        enqueue_workgroups(
            meta,
            pipeline,
            bind_group,
            (d1 + 7) / 8,
            (d2 + 7) / 8,
            1,
            #[cfg(feature = "wgpu_debug")]
            crate::wgpu::device::QueueDebugInfo::new(&format!("copy2d dtype:{:?}", dtype), d1 * d2),
        );
    }
    return Ok(());
}