use std::sync::Arc;


use wgpu::core::device;

use crate::{wgpu::{cache::BufferReference, device::Pipelines}, Layout, Shape, WgpuDevice};

use super::{create_bind_group_input2, enqueue_workgroups, get_meta};

pub fn queue_matmul_buffer1(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input1: Arc<BufferReference>,
    buffer_input2: Arc<BufferReference>,
    b: u32,
    m: u32,
    n: u32,
    k: u32,
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    let mut input1_stride = layout_input1.stride().iter().rev();
    let mut input2_stride = layout_input2.stride().iter().rev();

    let mut meta = get_meta(&dev);
    meta.add(b);
    meta.add(m);
    meta.add(k);
    meta.add(n);

    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_k
    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_m
    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_b
    meta.add(layout_input1.start_offset()); //input1_offset

    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_n
    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_k
    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_b
    meta.add(layout_input2.start_offset()); //input2_offset

    let pipeline = dev.get_pipeline(super::Shader::Matmul(dtype), Pipelines::MatmulBuffer)?;
  
    let bind_group = create_bind_group_input2(
        buffer_dest,
        buffer_input1,
        buffer_input2,
    );
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        (n  + 15) / 16,
        (m  + 15) / 16,
        b,
        (k * m * n) as usize,
        #[cfg(feature = "wgpu_debug")]
        crate::wgpu::device::QueueDebugInfo::new(&format!("matmul, dtype:{:?}", dtype)),
    );
    return Ok(());
}


//shader1b
pub fn queue_matmul_buffer1b(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input1: Arc<BufferReference>,
    buffer_input2: Arc<BufferReference>,
    b: u32,
    m: u32,
    n: u32,
    k: u32,
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    let mut input1_stride = layout_input1.stride().iter().rev();
    let mut input2_stride = layout_input2.stride().iter().rev();

    let mut meta = get_meta(&dev);
    meta.add(b);
    meta.add(m);
    meta.add(k);
    meta.add(n);

    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_k
    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_m
    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_b
    meta.add(layout_input1.start_offset()); //input1_offset

    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_n
    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_k
    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_b
    meta.add(layout_input2.start_offset()); //input2_offset

    let pipeline = dev.get_pipeline(super::Shader::Matmul(dtype), Pipelines::MatmulBuffer1b)?;
  
    let bind_group = create_bind_group_input2(
        buffer_dest,
        buffer_input1,
        buffer_input2,
    );
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        ((n + 15) / 16  + 15) / 16,
        (m  + 15) / 16,
        b,
        (k * m * n) as usize,
        #[cfg(feature = "wgpu_debug")]
        crate::wgpu::device::QueueDebugInfo::new(&format!("matmul, dtype:{:?}", dtype)),
    );
    return Ok(());
}


//shader3
pub fn queue_matmul_buffer3(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input1: Arc<BufferReference>,
    buffer_input2: Arc<BufferReference>,
    b: u32,
    m: u32,
    n: u32,
    k: u32,
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    let mut input1_stride = layout_input1.stride().iter().rev();
    let mut input2_stride = layout_input2.stride().iter().rev();

    let mut meta = get_meta(&dev);
    meta.add(b);
    meta.add(m);
    meta.add(k);
    meta.add(n);

    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_k
    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_m
    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_b
    meta.add(layout_input1.start_offset()); //input1_offset

    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_n
    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_k
    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_b
    meta.add(layout_input2.start_offset()); //input2_offset

    let pipeline = dev.get_pipeline(super::Shader::Matmul(dtype), Pipelines::Matmul3Buffer)?;
  
    let bind_group = create_bind_group_input2(
        buffer_dest,
        buffer_input1,
        buffer_input2,
    );
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        (n + 15) / 16,
        (m + 15) / 16,
        b,
        (k * m * n) as usize,
        #[cfg(feature = "wgpu_debug")]
        crate::wgpu::device::QueueDebugInfo::new(&format!("matmul3, dtype:{:?}", dtype)),
    );
    return Ok(());
}




//shader4
// pub fn queue_matmul_buffer4(
//     dev: &WgpuDevice,
//     buffer_dest: Arc<BufferReference>,
//     buffer_input1: Arc<BufferReference>,
//     buffer_input2: Arc<BufferReference>,
//     b: u32,
//     m: u32,
//     n: u32,
//     k: u32,
//     layout_input1: &Layout,
//     layout_input2: &Layout,
//     dtype: crate::DType,
// ) -> crate::Result<()> {
//     let mut input1_stride = layout_input1.stride().iter().rev();
//     let mut input2_stride = layout_input2.stride().iter().rev();

//     let input1_stride_k = *input1_stride.next().unwrap_or(&1);
//     let input1_stride_m = *input1_stride.next().unwrap_or(&1);
//     let input1_stride_b = *input1_stride.next().unwrap_or(&1);

//     let input2_stride_n = *input2_stride.next().unwrap_or(&1);
//     let input2_stride_k = *input2_stride.next().unwrap_or(&1);
//     let input2_stride_b = *input2_stride.next().unwrap_or(&1);

//     if n / 16 > 0 && m / 16 > 0{
//         let mut meta = get_meta(&dev);
//         meta.add(b);
//         meta.add(m);
//         meta.add(k);
//         meta.add(n);

//         //meta.add(input1_stride_k); //input1_stride_n
//         //meta.add(input1_stride_m); //input1_stride_m
//         meta.add(input1_stride_b); //input1_stride_b
//         meta.add(layout_input1.start_offset()); //input1_offset

//         //meta.add(input2_stride_n); //input2_stride_k
//         //meta.add(input2_stride_k); //input2_stride_n
//         meta.add(input2_stride_b); //input2_stride_b
//         meta.add(layout_input2.start_offset()); //input2_offset

//         let pipeline = dev.get_pipeline(super::Shader::Matmul(dtype), Pipelines::Matmul4Buffer)?;
    
//         let bind_group = create_bind_group_input2(
//             buffer_dest.clone(),
//             buffer_input1.clone(),
//             buffer_input2.clone(),
//         );
//         enqueue_workgroups(
//             meta,
//             pipeline,
//             bind_group,
//             n  / 16,
//             m  / 16,
//             b,
//             (k * m * n) as usize,
//             #[cfg(feature = "wgpu_debug")]
//             crate::wgpu::device::QueueDebugInfo::new(&format!("matmul4, dtype:{:?}", dtype)),
//         );
//     }
//     //calcualte rest:
//     if n % 16 != 0 || m % 16 != 0 {
//         let mut meta = get_meta(&dev);
//         meta.add(b);
//         meta.add(m);
//         meta.add(k);
//         meta.add(n);
    
//         //meta.add(input1_stride_k); //input1_stride_n
//         //meta.add(input1_stride_m); //input1_stride_m
//         meta.add(input1_stride_b); //input1_stride_b
//         meta.add(layout_input1.start_offset()); //input1_offset

//         //meta.add(input2_stride_n); //input2_stride_k
//         //meta.add(input2_stride_k); //input2_stride_n
//         meta.add(input2_stride_b); //input2_stride_b
//         meta.add(layout_input2.start_offset()); //input2_offset
//         meta.add((n  / 16) * 16); //xoffset
//         meta.add((m  / 16) * 16); //yoffset
    
//         let pipeline = dev.get_pipeline(super::Shader::Matmul(dtype), Pipelines::Matmul4endBuffer)?;
      
//         let bind_group = create_bind_group_input2(
//             buffer_dest,
//             buffer_input1,
//             buffer_input2,
//         );
//         enqueue_workgroups(
//             meta,
//             pipeline,
//             bind_group,
//             (n + 15)  / 16,
//             (m + 15) / 16,
//             b,
//             (k * m * n) as usize,
//             #[cfg(feature = "wgpu_debug")]
//             crate::wgpu::device::QueueDebugInfo::new(&format!("matmul4end, dtype:{:?}", dtype)),
//         );
//     }

//     return Ok(());
// }

fn round_up_to_nearest_16(m: u32) -> u32 {
    (m + 15) & !15
}

//shader5
pub fn queue_matmul_buffer(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input1: Arc<BufferReference>,
    buffer_input2: Arc<BufferReference>,
    b: u32,
    m: u32,
    n: u32,
    k: u32,
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    let mut input1_stride = layout_input1.stride().iter().rev();
    let mut input2_stride = layout_input2.stride().iter().rev();

    let input1_stride_k = *input1_stride.next().unwrap_or(&1);
    let input1_stride_m = *input1_stride.next().unwrap_or(&1);
    let input1_stride_b = *input1_stride.next().unwrap_or(&1);

    let input2_stride_n = *input2_stride.next().unwrap_or(&1);
    let input2_stride_k = *input2_stride.next().unwrap_or(&1);
    let input2_stride_b = *input2_stride.next().unwrap_or(&1);

    let const_vec = vec![input1_stride_k, input1_stride_m, input2_stride_n, input2_stride_k];
    
    //let use_linex;
    //let use_liney;

    //let use_matmul_end;
 //   if n / 16 > 0 && m / 16 > 0{
        let mut meta = get_meta(&dev);
        meta.add(b);
        meta.add(m);
        meta.add(n);
        meta.add(k);

        meta.add(input1_stride_b); //input1_stride_b
        meta.add(layout_input1.start_offset()); //input1_offset

        meta.add(input2_stride_b); //input2_stride_b
        meta.add(layout_input2.start_offset()); //input2_offset

        let pipeline = dev.get_pipeline_const(super::Shader::Matmul(dtype), Pipelines::Matmul7Buffer, const_vec.clone());
    
        let bind_group = create_bind_group_input2(
            buffer_dest.clone(),
            buffer_input1.clone(),
            buffer_input2.clone(),
        );
        enqueue_workgroups(
            meta,
            pipeline,
            bind_group,
           (n + 15) / 16,
           (m + 15) / 16,
            b,
            (k as usize * m as usize * n as usize),
            #[cfg(feature = "wgpu_debug")]
            crate::wgpu::device::QueueDebugInfo::new(&format!("matmul7, dtype:{:?}", dtype)),
        );

        //use_linex = m % 16 != 0;
        //use_liney = n % 16 != 0;
        return Ok(());
 //   }
 //   else{
        // let mut input1_stride = layout_input1.stride().iter().rev();
        // let mut input2_stride = layout_input2.stride().iter().rev();

        // let input1_stride_k = *input1_stride.next().unwrap_or(&1);
        // let input1_stride_m = *input1_stride.next().unwrap_or(&1);
        // let input1_stride_b = *input1_stride.next().unwrap_or(&1);

        // let input2_stride_n = *input2_stride.next().unwrap_or(&1);
        // let input2_stride_k = *input2_stride.next().unwrap_or(&1);
        // let input2_stride_b = *input2_stride.next().unwrap_or(&1);

        // let const_vec = vec![input1_stride_k, input1_stride_m, input2_stride_n, input2_stride_k];

        // let mut meta = get_meta(&dev);
        // meta.add(b);
        // meta.add(m);
        // meta.add(k);
        // meta.add(n);
    
        // //meta.add(input1_stride_k); //input1_stride_n
        // //meta.add(input1_stride_m); //input1_stride_m
        // meta.add(input1_stride_b); //input1_stride_b
        // meta.add(layout_input1.start_offset()); //input1_offset
    
        // //meta.add(input2_stride_n); //input2_stride_k
        // //meta.add(input2_stride_k); //input2_stride_n
        // meta.add(input2_stride_b); //input2_stride_b
        // meta.add(layout_input2.start_offset()); //input2_offset
        
        // let pipeline = dev.get_pipeline_const(super::Shader::Matmul(dtype), Pipelines::MatmulBuffer, const_vec);
    
        // let bind_group = create_bind_group_input2(
        //     buffer_dest,
        //     buffer_input1,
        //     buffer_input2,
        // );
        // enqueue_workgroups(
        //     meta,
        //     pipeline,
        //     bind_group,
        //     (n  + 15) / 16,
        //     (m  + 15) / 16,
        //     b,
        //     (k as usize * m as usize * n as usize),
        //     #[cfg(feature = "wgpu_debug")]
        //     crate::wgpu::device::QueueDebugInfo::new(&format!("matmul_end, dtype:{:?}", dtype)),
        // );    
        // return Ok(());
    //}
}



//shader5
pub fn queue_matmul_5_test_buffer(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input1: Arc<BufferReference>,
    buffer_input2: Arc<BufferReference>,
    b: u32,
    m: u32,
    n: u32,
    k: u32,
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    //let mut input1_stride = layout_input1.stride().iter().rev();
    //let mut input2_stride = layout_input2.stride().iter().rev();

    

    //let use_linex;
    //let use_liney;

    //let use_matmul_end;
    if n / 16 > 0 && m / 16 > 0{

        let new_m = round_up_to_nearest_16(m);
        let new_n = round_up_to_nearest_16(n);
        let new_k = round_up_to_nearest_16(k);
        //let new_k = k;

        let need_different_output_buffer = new_m != m || new_n != n;

        let (buffer_input1_padded, layout_input1_padded) = 
            if m % 16 == 0{
                (buffer_input1, layout_input1.clone())
            }
            else{
                //println!("pad input buffer 1:");
                let buffer_input1_padded = BufferReference::new(&dev, b * (new_m * new_k) * 4);   
                
                let dest_layout = crate::Layout::contiguous(&Shape::from((b as usize, new_m as usize, new_k as usize)));
                super::queue_copy3d(dev, buffer_input1_padded.clone(), buffer_input1, dtype, layout_input1, (b, m, k), &dest_layout)?;
                (buffer_input1_padded, dest_layout)
            };

        let (buffer_input2_padded, layout_input2_padded) = 
            if n % 16 == 0{
                (buffer_input2, layout_input2.clone())
            }
            else{
                //println!("pad input buffer 2:");
                let buffer_input2_padded = BufferReference::new(&dev, b * (new_k * new_n) * 4);   
                
                let dest_layout = crate::Layout::contiguous(&Shape::from((b as usize, new_k as usize, new_n as usize)));
                super::queue_copy3d(dev, buffer_input2_padded.clone(), buffer_input2, dtype, layout_input2, (b, k, n),&dest_layout)?;
                (buffer_input2_padded, dest_layout)
            };
        
        let buffer_dest_padded = 
            if need_different_output_buffer{
                let buffer_dest = BufferReference::new(&dev, b * (new_m * new_n) * 4);   
                buffer_dest
            }
            else{
                buffer_dest.clone()
            };



        let mut input1_stride = layout_input1_padded.stride().iter().rev();
        let mut input2_stride = layout_input2_padded.stride().iter().rev();

        let input1_stride_k = *input1_stride.next().unwrap_or(&1);
        let input1_stride_m = *input1_stride.next().unwrap_or(&1);
        let input1_stride_b = *input1_stride.next().unwrap_or(&1);

        let input2_stride_n = *input2_stride.next().unwrap_or(&1);
        let input2_stride_k = *input2_stride.next().unwrap_or(&1);
        let input2_stride_b = *input2_stride.next().unwrap_or(&1);

        let const_vec = vec![input1_stride_k, input1_stride_m, input2_stride_n, input2_stride_k];
        
        //println!("InputStride1: {:?}", layout_input1_padded.stride());
        //println!("InputStride2: {:?}", layout_input2_padded.stride());

        let mut meta = get_meta(&dev);
        meta.add(b);
        meta.add(new_m);
        meta.add(new_n);
        meta.add(new_k);

        //meta.add(input1_stride_k); //input1_stride_k
        //meta.add(input1_stride_m); //input1_stride_m
        meta.add(input1_stride_b); //input1_stride_b
        meta.add(layout_input1_padded.start_offset()); //input1_offset

        //meta.add(input2_stride_n); //input2_stride_n
        //meta.add(input2_stride_k); //input2_stride_k
        meta.add(input2_stride_b); //input2_stride_b
        meta.add(layout_input2_padded.start_offset()); //input2_offset

        let pipeline = dev.get_pipeline_const(super::Shader::Matmul(dtype), Pipelines::Matmul5Buffer, const_vec.clone());
    
        let bind_group = create_bind_group_input2(
            buffer_dest_padded.clone(),
            buffer_input1_padded.clone(),
            buffer_input2_padded.clone(),
        );
        enqueue_workgroups(
            meta,
            pipeline,
            bind_group,
           (new_n ) / 16,
           (new_m ) / 16,
            b,
            (k as usize * m as usize * n as usize),
            #[cfg(feature = "wgpu_debug")]
            crate::wgpu::device::QueueDebugInfo::new(&format!("matmul5, dtype:{:?}", dtype)),
        );

        if need_different_output_buffer{
            //println!("copy padding dest to dest");
            let dest_padding_layout = crate::Layout::contiguous(&Shape::from((b as usize, new_m as usize, new_n as usize)));
            let dest_layout = crate::Layout::contiguous(&Shape::from((b as usize, m as usize, n as usize)));
            super::queue_copy3d(dev, buffer_dest, buffer_dest_padded, dtype, &dest_padding_layout, (b, m, n), &dest_layout)?;
        }

        //use_linex = m % 16 != 0;
        //use_liney = n % 16 != 0;
        return Ok(());
    }
    else{
        let mut input1_stride = layout_input1.stride().iter().rev();
        let mut input2_stride = layout_input2.stride().iter().rev();

        let input1_stride_k = *input1_stride.next().unwrap_or(&1);
        let input1_stride_m = *input1_stride.next().unwrap_or(&1);
        let input1_stride_b = *input1_stride.next().unwrap_or(&1);

        let input2_stride_n = *input2_stride.next().unwrap_or(&1);
        let input2_stride_k = *input2_stride.next().unwrap_or(&1);
        let input2_stride_b = *input2_stride.next().unwrap_or(&1);

        let const_vec = vec![input1_stride_k, input1_stride_m, input2_stride_n, input2_stride_k];

        let mut meta = get_meta(&dev);
        meta.add(b);
        meta.add(m);
        meta.add(k);
        meta.add(n);
    
        //meta.add(input1_stride_k); //input1_stride_n
        //meta.add(input1_stride_m); //input1_stride_m
        meta.add(input1_stride_b); //input1_stride_b
        meta.add(layout_input1.start_offset()); //input1_offset
    
        //meta.add(input2_stride_n); //input2_stride_k
        //meta.add(input2_stride_k); //input2_stride_n
        meta.add(input2_stride_b); //input2_stride_b
        meta.add(layout_input2.start_offset()); //input2_offset
        
        let pipeline = dev.get_pipeline_const(super::Shader::Matmul(dtype), Pipelines::MatmulBuffer, const_vec);
    
        let bind_group = create_bind_group_input2(
            buffer_dest,
            buffer_input1,
            buffer_input2,
        );
        enqueue_workgroups(
            meta,
            pipeline,
            bind_group,
            (n  + 15) / 16,
            (m  + 15) / 16,
            b,
            (k as usize * m as usize * n as usize),
            #[cfg(feature = "wgpu_debug")]
            crate::wgpu::device::QueueDebugInfo::new(&format!("matmul_end, dtype:{:?}", dtype)),
        );    
        return Ok(());
    }
    // else{ //end useses complete matrix, use the shader, that would be called less often.
    //     //use_matmul_end = true;
    //     use_linex = false;
    //     use_liney = false;
    //     // if m <= n{
    //     //     use_linex = true;
    //     //     use_liney = false;
    //     // }
    //     // else{
    //     //     use_linex = false;
    //     //     use_liney = true;
    //     // }
    // }
   
    // if use_matmul_end{
    // if use_linex || use_liney{
    //     let mut meta = get_meta(&dev);
    //     meta.add(b);
    //     meta.add(m);
    //     meta.add(k);
    //     meta.add(n);
    
    //     //meta.add(input1_stride_k); //input1_stride_n
    //     //meta.add(input1_stride_m); //input1_stride_m
    //     meta.add(input1_stride_b); //input1_stride_b
    //     meta.add(layout_input1.start_offset()); //input1_offset
    
    //     //meta.add(input2_stride_n); //input2_stride_k
    //     //meta.add(input2_stride_k); //input2_stride_n
    //     meta.add(input2_stride_b); //input2_stride_b
    //     meta.add(layout_input2.start_offset()); //input2_offset
    //     meta.add((n  / 16) * 16); //xoffset
    //     meta.add((m  / 16) * 16); //xoffset
        
    //     let pipeline = dev.get_pipeline_const(super::Shader::Matmul(dtype), Pipelines::Matmul1endBuffer, const_vec);
    
    //     let bind_group = create_bind_group_input2(
    //         buffer_dest,
    //         buffer_input1,
    //         buffer_input2,
    //     );
    //     enqueue_workgroups(
    //         meta,
    //         pipeline,
    //         bind_group,
    //         (n  + 15) / 16,
    //         (m  + 15) / 16,
    //         b,
    //         (k * (m * n - ((m  / 16) * 16) * ((n  / 16) * 16))) as usize,
    //         #[cfg(feature = "wgpu_debug")]
    //         crate::wgpu::device::QueueDebugInfo::new(&format!("matmul_end, dtype:{:?}", dtype)),
    //     );
    // }
       
       
    // // }
    // // else{
    //     //calcualte rest:
    //     if use_linex{
    //         let mut meta = get_meta(&dev);
    //         meta.add(b);
    //         meta.add(m);
    //         meta.add(k);
    //         meta.add(n);

    //         //meta.add(input1_stride_k); //input1_stride_n
    //         //meta.add(input1_stride_m); //input1_stride_m
    //         meta.add(input1_stride_b); //input1_stride_b
    //         meta.add(layout_input1.start_offset()); //input1_offset

    //         //meta.add(input2_stride_n); //input2_stride_k
    //         //meta.add(input2_stride_k); //input2_stride_n
    //         meta.add(input2_stride_b); //input2_stride_b
    //         meta.add(layout_input2.start_offset()); //input2_offset
    //         meta.add(0u32); //xoffset
    //         meta.add((m  / 16) * 16); //yoffset

    //         let pipeline = dev.get_pipeline(super::Shader::Matmul(dtype), Pipelines::Matmul7Buffer)?;
           
    //         let bind_group = create_bind_group_input2(
    //             buffer_dest.clone(),
    //             buffer_input1.clone(),
    //             buffer_input2.clone(),
    //         );
    //         #[cfg(feature = "wgpu_debug")]
    //         let debug =  crate::wgpu::device::QueueDebugInfo::new(&format!("matmul_linex, dtype:{:?}, pipeline {:?}, n: {n}, m : {}/{m}, k : {k}, b: {b}", dtype, pipeline, m % 16));
    //         enqueue_workgroups(
    //             meta,
    //             pipeline,
    //             bind_group,
    //             (n + 255)  / 256,
    //             m % 16,
    //             b,
    //             (k * (m % 16) * n) as usize,
    //             #[cfg(feature = "wgpu_debug")]
    //             debug
    //         );
    //     }
    //     //calcualte rest:
    //     if use_liney {
    //         let mut meta = get_meta(&dev);
    //         meta.add(b);
    //         meta.add(m);
    //         meta.add(k);
    //         meta.add(n);

    //         //meta.add(input1_stride_k); //input1_stride_n
    //         //meta.add(input1_stride_m); //input1_stride_m
    //         meta.add(input1_stride_b); //input1_stride_b
    //         meta.add(layout_input1.start_offset()); //input1_offset

    //         //meta.add(input2_stride_n); //input2_stride_k
    //         //meta.add(input2_stride_k); //input2_stride_n
    //         meta.add(input2_stride_b); //input2_stride_b
    //         meta.add(layout_input2.start_offset()); //input2_offset
    //         meta.add((n  / 16) * 16); //xoffset
    //         meta.add(0u32); //yoffset
            
    //         let pipeline = dev.get_pipeline(super::Shader::Matmul(dtype), Pipelines::Matmul7Buffer)?;
           
    //         let bind_group = create_bind_group_input2(
    //             buffer_dest,
    //             buffer_input1,
    //             buffer_input2,
    //         );

    //         #[cfg(feature = "wgpu_debug")]
    //         let debug =  crate::wgpu::device::QueueDebugInfo::new(&format!("matmul_liney, dtype:{:?}, pipeline {:?}", dtype, pipeline));
    //         enqueue_workgroups(
    //             meta,
    //             pipeline,
    //             bind_group,
    //             n % 16,
    //             (m + 255) / 256,
    //             b,
    //             (k * (n % 16) * m) as usize,
    //             #[cfg(feature = "wgpu_debug")]
    //             debug
    //         );
    //     }

    // }
    
    return Ok(());
}




//shader6
// pub fn queue_matmul_buffer6(
//     dev: &WgpuDevice,
//     buffer_dest: Arc<BufferReference>,
//     buffer_input1: Arc<BufferReference>,
//     buffer_input2: Arc<BufferReference>,
//     b: u32,
//     m: u32,
//     n: u32,
//     k: u32,
//     layout_input1: &Layout,
//     layout_input2: &Layout,
//     dtype: crate::DType,
// ) -> crate::Result<()> {
//     let mut input1_stride = layout_input1.stride().iter().rev();
//     let mut input2_stride = layout_input2.stride().iter().rev();

//     let input1_stride_k = *input1_stride.next().unwrap_or(&1);
//     let input1_stride_m = *input1_stride.next().unwrap_or(&1);
//     let input1_stride_b = *input1_stride.next().unwrap_or(&1);

//     let input2_stride_k = *input2_stride.next().unwrap_or(&1);
//     let input2_stride_n = *input2_stride.next().unwrap_or(&1);
//     let input2_stride_b = *input2_stride.next().unwrap_or(&1);

//     if n / 16 > 0 && m / 16 > 0{
//         let mut meta = get_meta(&dev);
//         meta.add(b);
//         meta.add(m);
//         meta.add(k);
//         meta.add(n);

//         //meta.add(input1_stride_k); //input1_stride_n
//         //meta.add(input1_stride_m); //input1_stride_m
//         meta.add(input1_stride_b); //input1_stride_b
//         meta.add(layout_input1.start_offset()); //input1_offset

//         //meta.add(input2_stride_n); //input2_stride_k
//         //meta.add(input2_stride_k); //input2_stride_n
//         meta.add(input2_stride_b); //input2_stride_b
//         meta.add(layout_input2.start_offset()); //input2_offset

//         let pipeline = dev.get_pipeline(super::Shader::Matmul(dtype), Pipelines::Matmul6Buffer)?;
    
//         let bind_group = create_bind_group_input2(
//             buffer_dest.clone(),
//             buffer_input1.clone(),
//             buffer_input2.clone(),
//         );
//         enqueue_workgroups(
//             meta,
//             pipeline,
//             bind_group,
//             n  / 16,
//             m  / 16,
//             b,
//             (k * m * n) as usize,
//             #[cfg(feature = "wgpu_debug")]
//             crate::wgpu::device::QueueDebugInfo::new(&format!("matmul6, dtype:{:?}", dtype)),
//         );
//     }
//     //calcualte rest:
//     if n % 16 != 0 || m % 16 != 0 {
//         let mut meta = get_meta(&dev);
//         meta.add(b);
//         meta.add(m);
//         meta.add(k);
//         meta.add(n);
    
//         //meta.add(input1_stride_k); //input1_stride_n
//         //meta.add(input1_stride_m); //input1_stride_m
//         meta.add(input1_stride_b); //input1_stride_b
//         meta.add(layout_input1.start_offset()); //input1_offset

//         //meta.add(input2_stride_n); //input2_stride_k
//         //meta.add(input2_stride_k); //input2_stride_n
//         meta.add(input2_stride_b); //input2_stride_b
//         meta.add(layout_input2.start_offset()); //input2_offset
//         meta.add((n  / 16) * 16); //xoffset
//         meta.add((m  / 16) * 16); //xoffset
    
//         let pipeline = dev.get_pipeline(super::Shader::Matmul(dtype), Pipelines::Matmul1endBuffer)?;
      
//         let bind_group = create_bind_group_input2(
//             buffer_dest,
//             buffer_input1,
//             buffer_input2,
//         );
//         enqueue_workgroups(
//             meta,
//             pipeline,
//             bind_group,
//             (n + 15)  / 16,
//             (m + 15) / 16,
//             b,
//             (k * m * n) as usize,
//             #[cfg(feature = "wgpu_debug")]
//             crate::wgpu::device::QueueDebugInfo::new(&format!("matmul4end, dtype:{:?}", dtype)),
//         );
//     }

//     return Ok(());
// }
