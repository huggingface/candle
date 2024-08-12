use candle_wgpu_kernels::copy::Functions;

use super::*;

pub fn queue_copy_strided(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    dtype: crate::DType,
    input_layout: &crate::Layout,
    dst_offset: u32,
) -> crate::Result<()> {
    if input_layout.shape().elem_count() > 0{
        let result = input_layout.shape().dims().iter().zip(input_layout.stride()).filter(|(dim, _)| **dim > 1).map(|(dim, stride)| (*dim, *stride)).collect::<Vec<(usize,usize)>>();
        let (shape, stride) : (Vec<usize>, Vec<usize>) = result.into_iter().unzip();
        if shape.len() == 3{ //try copy 3d
            if dst_offset == 0{
                let layout: Layout = Layout::new(crate::Shape::from_dims(&shape), stride, input_layout.start_offset());
                return queue_copy3d(dev, buffer_dest, buffer_input, dtype, &layout, (shape[0] as u32, shape[1] as u32, shape[2] as u32), &Layout::contiguous(shape));
            }
        }

        let mut meta = get_meta(&dev);
        meta.add(dst_offset);
        meta.add_layout1(&input_layout);
    
        let pipeline = meta.get_pipeline(Pipelines::Copy(get_dtype(dtype)?, Functions::CopyStrided));
    
        let bind_group = create_bind_group_input1( buffer_dest, buffer_input);
        enqueue_big_extra(
            meta,
            pipeline,
            bind_group,
            input_layout.shape().elem_count() as u32,
            #[cfg(feature="wgpu_debug")]
            Some(format!("shape: {:?}, stride: {:?}", input_layout.shape(), input_layout.stride()))
        );
    }
    return Ok(());
}


//This is ~30% faster than using a shader to copy, but a shader dispatch call can be easier cached. therefore we just use the slower copy function at the moment.
//In addition the copy is often not the bottle neck(but matmul or conv-dispatch call)
// pub fn queue_copy_old(
//     dev: &WgpuDevice,
//     buffer_dest: BufferReferenceId,
//     buffer_input: BufferReferenceId,
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
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    destination_offset: usize,
    source_offset: usize,
    copy_size: usize,
    dtype : crate::DType
) -> crate::Result<()> {
    if copy_size > 0{
        let const_vec = vec![
            (source_offset == 0) as u32, 
            (destination_offset == 0) as u32];
        
        let mut meta = get_meta(&dev);
        // if copy_size % 4 == 0 && source_offset % 4 == 0 && destination_offset % 4 == 0{
        //     meta.add(copy_size/4);
        //     meta.add(destination_offset/4);
        //     meta.add(source_offset/4);
            
        //     let inplaceble = OpIsInplaceable{ input1_inplaceable:  destination_offset == source_offset, input2_inplaceable: false };
        //     let pipeline = meta.get_pipeline_const_inplace(Pipelines::Copy(get_dtype(dtype)?, Functions::Copy4), const_vec, inplaceble);
        
        //     let bind_group = create_bind_group_input1_16( buffer_dest, buffer_input);
        //     enqueue_big(
        //         meta,
        //         pipeline,
        //         bind_group,
        //         (copy_size/4) as u32
        //     );
        // }
        // else{
            meta.add(copy_size);
            meta.add(destination_offset);
            meta.add(source_offset);
            
            let inplaceble = OpIsInplaceable{ input1_inplaceable:  destination_offset == source_offset, input2_inplaceable: false };
            let pipeline = meta.get_pipeline_const_inplace(Pipelines::Copy(get_dtype(dtype)?, Functions::Copy),const_vec, inplaceble);
        
            let bind_group = create_bind_group_input1( buffer_dest, buffer_input);
            enqueue_big(
                meta,
                pipeline,
                bind_group,
                copy_size as u32
            );
        //}

    }
    return Ok(());
}




pub fn queue_copy2d(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    dtype: crate::DType,
    d1: u32,
    d2: u32,
    input_stride1: u32,
    dest_stride1: u32,
    input_offset: u32,
    dest_offset: u32,
) -> crate::Result<()> {
    if buffer_dest.size > 0 && buffer_input.size > 0{
        if d1 == 1 || (input_stride1 == d2 && input_stride1 == dest_stride1){
            return queue_copy(dev, buffer_dest, buffer_input, dest_offset as usize, input_offset as usize, (d2 * d1) as usize, dtype);
        }
        let const_vec = vec![
            input_offset, 
            (dest_offset == 0) as u32];
       
        let mut meta = get_meta(&dev);
        meta.add(d1);
        meta.add(d2);
        meta.add(input_stride1);
        meta.add(dest_stride1);
        if dest_offset != 0 {
            meta.add(dest_offset);
        }
     
        let bind_group = create_bind_group_input1( buffer_dest, buffer_input);
        
        let x = (d1 + 15) / 16;
        let y = (d2 + 15) / 16;


        if y > MAX_DISPATCH_SIZE{
            let pipeline = meta.get_pipeline_const(Pipelines::Copy(get_dtype(dtype)?, Functions::Copy2dTranspose), const_vec);
            enqueue_workgroups(
                meta,
                pipeline,
                bind_group,
                y.min(65535),
                x,
                (y + 65534) / 65535,
                (d1 * d2) as usize,
            );
        }
        else{
            let pipeline = meta.get_pipeline_const(Pipelines::Copy(get_dtype(dtype)?, Functions::Copy2d), const_vec);
            enqueue_workgroups(
                meta,
                pipeline,
                bind_group,
                x.min(65535),
                y,
                (x + 65534) / 65535,
                (d1 * d2) as usize
            );
        }
    }
    return Ok(());
}

pub fn queue_copy3d(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
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

        let const_vec = vec![
            input_layout.start_offset(),
            (dest_stride_1 != 1) as usize,
            (dest_stride_2 != 1) as usize,
            (dest_stride_3 != 1) as usize,
            
            (input1_stride_1 != 1) as usize,
            (input1_stride_2 != 1) as usize,
            (input1_stride_3 != 1) as usize,            
            ];

        let mut meta = get_meta(&dev);
        meta.add(input_shape.2);
        meta.add(input_shape.1);
        meta.add(dest_stride_1);
        meta.add(dest_stride_2);
        meta.add(dest_stride_3);
        meta.add(input1_stride_1);
        meta.add(input1_stride_2);
        meta.add(input1_stride_3);
        
        let bind_group = create_bind_group_input1( buffer_dest, buffer_input);

        let pipeline = meta.get_pipeline_const(Pipelines::Copy(get_dtype(dtype)?, Functions::Copy3d), const_vec);
        enqueue_workgroups(
            meta,
            pipeline,
            bind_group,
            (input_shape.2 + 15 ) / 16 as u32,
            (input_shape.1 + 15)  / 16 as u32,
            input_shape.0 as u32,
            input_layout.shape().elem_count()
        );
    }
    return Ok(());
}


pub fn queue_copy3d_padded(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
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

        let dest_shape = dest_layout.shape().dims3()?;
        
        let const_vec = vec![
            input_layout.start_offset(),
            (dest_stride_1 != 1) as usize,
            (dest_stride_2 != 1) as usize,
            (dest_stride_3 != 1) as usize,
            
            (input1_stride_1 != 1) as usize,
            (input1_stride_2 != 1) as usize,
            (input1_stride_3 != 1) as usize,            
            ];
        
        let mut meta = get_meta(&dev);
        meta.add(input_shape.2);
        meta.add(input_shape.1);
        meta.add(dest_stride_1);
        meta.add(dest_stride_2);
        meta.add(dest_stride_3);
        meta.add(input1_stride_1);
        meta.add(input1_stride_2);
        meta.add(input1_stride_3);
        meta.add(dest_shape.2);
        meta.add(dest_shape.1);


        let bind_group = create_bind_group_input1( buffer_dest, buffer_input);
        let pipeline = if input_shape.0 == 1{
            Functions::Copy3dPaddedNobatch
        } 
        else{
            Functions::Copy3dPadded
        };
        let pipeline = meta.get_pipeline_const(Pipelines::Copy(get_dtype(dtype)?, pipeline), const_vec);
        enqueue_workgroups(
            meta,
            pipeline,
            bind_group,
            ((dest_shape.2 + 15) / 16) as u32,
            ((dest_shape.1 + 15) / 16) as u32,
            input_shape.0 as u32,
            input_layout.shape().elem_count(),
        );
    }
    return Ok(());
}