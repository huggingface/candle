use std::sync::Arc;


use crate::{wgpu::{cache::BufferReference, device::Pipelines}, WgpuDevice};

use super::{create_bind_group_input1, enqueue_workgroups, get_meta};

pub fn queue_max_pool2d(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input1: Arc<BufferReference>,
    layout: &crate::Layout,
    dtype: crate::DType,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> crate::Result<()> {
    
    let (b, c, h, w) = layout.shape().dims4()?;
    let h_out = (h - kernel_size.1) / stride.1 + 1;
    let w_out = (w - kernel_size.0) / stride.0 + 1;
    
    let input_stride = layout.stride();
    

    let mut meta = get_meta(&dev);

    meta.add(b);
    meta.add(c);
    meta.add(kernel_size.1);
    meta.add(kernel_size.0);
    meta.add(w);   //size_in_x
    meta.add(h);   //size_in_y
    meta.add(w_out * h_out * c); //Stride_batch_out
    meta.add(w_out * h_out); //stride_c_out
    meta.add(w_out); //stride_y_out
    meta.add(h_out); //size_y_out

    meta.add(input_stride[0]); //stride_batch_input
    meta.add(input_stride[1]); //stride_c_in
    meta.add(input_stride[2]); //stride_y_in
    meta.add(input_stride[3]); //stride_x_in
    meta.add(stride.1);
    meta.add(stride.0);
    meta.add(layout.start_offset());

    let pipeline = dev.get_pipeline(super::Shader::Pool2d(dtype), Pipelines::MaxPool2d)?;

    let bind_group = create_bind_group_input1(
        buffer_dest,
        buffer_input1,
    );
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        (w_out as u32 + 7) / 8,
        (h_out as u32 + 7) / 8,
        c as u32,
        #[cfg(feature = "wgpu_debug")] 
        crate::wgpu::device::QueueDebugInfo::new(&format!("max_pool2d, dtype:{:?}", dtype),h_out * w_out * b * c ),
    );
    return Ok(());
}



pub fn queue_avg_pool2d(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input1: Arc<BufferReference>,
    layout: &crate::Layout,
    dtype: crate::DType,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> crate::Result<()> {
    
    let (b, c, h, w) = layout.shape().dims4()?;
    let h_out = (h - kernel_size.1) / stride.1 + 1;
    let w_out = (w - kernel_size.0) / stride.0 + 1;
    
    let input_stride = layout.stride();
    

    let mut meta = get_meta(&dev);

    meta.add(b);
    meta.add(c);
    meta.add(kernel_size.1);
    meta.add(kernel_size.0);
    meta.add(w);   //size_in_x
    meta.add(h);   //size_in_y
    meta.add(w_out * h_out * c); //Stride_batch_out
    meta.add(w_out * h_out); //stride_c_out
    meta.add(w_out); //stride_y_out
    meta.add(h_out); //size_y_out

    meta.add(input_stride[0]); //stride_batch_input
    meta.add(input_stride[1]); //stride_c_in
    meta.add(input_stride[2]); //stride_y_in
    meta.add(input_stride[3]); //stride_x_in
    meta.add(stride.1);
    meta.add(stride.0);
    meta.add(layout.start_offset());

    let pipeline = dev.get_pipeline(super::Shader::Pool2d(dtype), Pipelines::AvgPool2d)?;

    let bind_group = create_bind_group_input1(
        buffer_dest,
        buffer_input1,
    );
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        (w_out as u32 + 7) / 8,
        (h_out as u32 + 7) / 8,
        c as u32,
        #[cfg(feature = "wgpu_debug")] 
        crate::wgpu::device::QueueDebugInfo::new(&format!("avg_pool2d, dtype:{:?}", dtype),w_out * h_out * c * b),
    );
    return Ok(());
}
