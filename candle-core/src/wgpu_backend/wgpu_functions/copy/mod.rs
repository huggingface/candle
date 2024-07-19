use std::{alloc::Layout, sync::Arc};

use crate::{wgpu::{cache::BufferReference, device::Pipelines}, WgpuDevice};

use super::{create_bind_group_input1, enqueue_big, enqueue_workgroups, get_meta, MAX_DISPATCH_SIZE};

pub fn queue_copy_strided(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input: Arc<BufferReference>,
    dtype: crate::DType,
    input_layout: &crate::Layout,
    dst_offset: u32,
) -> crate::Result<()> {
    if input_layout.shape().elem_count() > 0{
        let mut meta = get_meta(&dev);
        meta.add(dst_offset);
        meta.add_layout(&input_layout);
    
        //println!("queue_copy_strided: dst_offset: {dst_offset}, input_layout: {:?}", input_layout);

        let pipeline = dev.get_pipeline(super::Shader::Copy(dtype), Pipelines::CopyStrided)?;
    
        let bind_group = create_bind_group_input1( buffer_dest, buffer_input);
        enqueue_big(
            meta,
            pipeline,
            bind_group,
            input_layout.shape().elem_count() as u32,
            #[cfg(feature = "wgpu_debug")] 
            crate::wgpu::device::QueueDebugInfo::new(&format!("copy strided dtype:{:?}", dtype)),
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
        let mut meta = get_meta(&dev);
        meta.add(copy_size);
        meta.add(destination_offset);
        meta.add(source_offset);
        
        let pipeline = dev.get_pipeline(super::Shader::Copy(dtype), Pipelines::Copy)?;
    
        let bind_group = create_bind_group_input1( buffer_dest, buffer_input);
        enqueue_big(
            meta,
            pipeline,
            bind_group,
            copy_size as u32,
            #[cfg(feature = "wgpu_debug")] 
            crate::wgpu::device::QueueDebugInfo::new(&format!("copy strided dtype:{:?}", dtype)),
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

        let const_vec = vec![
            input_offset,
            dest_offset];
        
        let mut meta = get_meta(&dev);
        meta.add(d1);
        meta.add(d2);
        meta.add(input_stride1);
        meta.add(dest_stride1);

    
        let bind_group = create_bind_group_input1( buffer_dest, buffer_input);
        
        let x = (d1 + 15) / 16;
        let y = (((d2 + 7) / 8) + 15) / 16;


        if y > MAX_DISPATCH_SIZE{
            let pipeline = dev.get_pipeline_const(super::Shader::Copy(dtype), Pipelines::Copy2dTranspose, const_vec);
            enqueue_workgroups(
                meta,
                pipeline,
                bind_group,
                y.min(65535),
                x,
                (y + 65534) / 65535,
                (d1 * d2) as usize,
                #[cfg(feature = "wgpu_debug")]
                crate::wgpu::device::QueueDebugInfo::new(&format!("copy2dTranpose d1: {d1}, d2: {d2}, input_stride1: {input_stride1}, dest_stride1:{dest_stride1},  dtype:{:?}", dtype)),
            );
        }
        else{
            let pipeline = dev.get_pipeline_const(super::Shader::Copy(dtype), Pipelines::Copy2d,const_vec);
            enqueue_workgroups(
                meta,
                pipeline,
                bind_group,
                x.min(65535),
                y,
                (x + 65534) / 65535,
                (d1 * d2) as usize,
                #[cfg(feature = "wgpu_debug")]
                crate::wgpu::device::QueueDebugInfo::new(&format!("copy2d d1: {d1}, d2: {d2}, dtype:{:?}", dtype)),
            );
        }
    }
    return Ok(());
}

pub fn queue_copy3d(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input: Arc<BufferReference>,
    dtype: crate::DType,
    input_layout: &crate::Layout,
    input_shape : (u32, u32, u32), //b, m, k
    dest_layout : &crate::Layout,
) -> crate::Result<()> {
    if buffer_dest.size > 0 && buffer_input.size > 0{
        let mut input1_stride = input_layout.stride().iter().rev();

    
        let input1_stride_1 = *input1_stride.next().unwrap_or(&1); //k
        let input1_stride_2 = *input1_stride.next().unwrap_or(&1); //m
        let input1_stride_3 = *input1_stride.next().unwrap_or(&1); //b
        
        let mut dest_stride = dest_layout.stride().iter().rev();
        let dest_stride_1 = *dest_stride.next().unwrap_or(&1);
        let dest_stride_2 = *dest_stride.next().unwrap_or(&1);
        let dest_stride_3 = *dest_stride.next().unwrap_or(&1);

        // if input_shape.0 == 1 && input1_stride_1 == 1 && dest_stride_1 == 1  { //batch == 1
        //     return queue_copy2d(dev, buffer_dest, buffer_input, dtype, input_shape.1, input_shape.2, input1_stride_2  as u32, dest_stride_2 as u32, input_layout.start_offset() as u32, 0)
        // }
        // else{
            let dest_shape = dest_layout.shape().dims3()?;

            let const_vec = vec![
                input_layout.start_offset(),
                dest_stride_1,
                dest_stride_2,
                dest_stride_3,
                input1_stride_1,
                input1_stride_2,
                input1_stride_3,
                ];
            
            let mut meta = get_meta(&dev);
            meta.add(input_shape.2);
            meta.add(input_shape.1);
            //meta.add(input1_stride_2);
            //meta.add(input1_stride_3);
    
            //meta.add(dest_stride_2);
            //meta.add(dest_stride_3);
    
        
            let bind_group = create_bind_group_input1( buffer_dest, buffer_input);
       
            let pipeline = dev.get_pipeline_const(super::Shader::Copy(dtype), Pipelines::Copy3d,const_vec);
            //println!("copy3d d1: {}, d2: {}, d3: {}, dtype:{:?}, input1_stride: {:?}, dest_stride: {:?}",input_shape.2, input_shape.1, input_shape.0, dtype,input_layout.stride(), dest_layout.stride());
            enqueue_workgroups(
                meta,
                pipeline,
                bind_group,
                (input_shape.2 + 15 ) / 16 as u32,
                (input_shape.1 + 15)  / 16 as u32,
                input_shape.0 as u32,
                input_layout.shape().elem_count(),
                #[cfg(feature = "wgpu_debug")]
                crate::wgpu::device::QueueDebugInfo::new(&format!("copy3d d1: {}, d2: {}, d3: {}, dtype:{:?}",input_shape.2, input_shape.1, input_shape.0, dtype)),
            );
        //}
       
        
    }
    return Ok(());
}