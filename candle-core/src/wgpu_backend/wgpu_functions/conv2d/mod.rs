use crate::wgpu::{device::Pipelines, BufferReferenceId, WgpuDevice};

use super::{create_bind_group_input2, enqueue_workgroups, get_meta};

pub fn queue_conv2d(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    dtype: crate::DType,
    params: &crate::conv::ParamsConv2D,
    input_layout: &crate::Layout,
    kernel_layout: &crate::Layout,
) -> crate::Result<()> {
    let input_stride = input_layout.stride();
    let kernel_stride = kernel_layout.stride();
    
    let (mut meta,  meta_offset) = get_meta(&dev, 24);

    meta.add(params.b_size);
    meta.add(params.c_in);
    meta.add(params.k_w);
    meta.add(params.k_h);
    meta.add(kernel_stride[3]); //kernel_x_stride
    meta.add(kernel_stride[2]); //kernel_y_stride
    meta.add(kernel_stride[1]); //kernel_c_stride
    meta.add(kernel_stride[0]); //kernel_b_stride
    meta.add(kernel_layout.start_offset());
    meta.add(params.i_w);   //size_in_x
    meta.add(params.i_h);   //size_in_y
    meta.add(params.out_w() * params.out_h() * params.c_out); //Stride_batch_out
    meta.add(params.out_w() * params.out_h()); //stride_c_out
    meta.add(params.out_w()); //stride_y_out
    meta.add(params.out_h()); //size_y_out

    meta.add(input_stride[0]); //stride_batch_input
    meta.add(input_stride[1]); //stride_c_in
    meta.add(input_stride[2]); //stride_y_in
    meta.add(input_stride[3]); //stride_x_in
    meta.add(params.padding);
    meta.add(params.stride);
    meta.add(params.dilation);
    meta.add(input_layout.start_offset());

    let pipeline = dev.get_pipeline(super::Shader::Conv2D(dtype), Pipelines::Conv2D)?;

    let bind_group = create_bind_group_input2(
        dev,
        pipeline.clone(),
        meta_offset,
        buffer_dest,
        buffer_input1,
        buffer_input2,
    );
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        (params.out_w() as u32 + 15) / 16,
        (params.out_h() as u32 + 16) / 16,
        params.c_out as u32,
        #[cfg(feature = "wgpu_debug")] 
        crate::wgpu::device::QueueDebugInfo::new(&format!("conv2d, kernel:{:?} dtype:{:?}", dtype, kernel_layout.shape()), params.out_w() * params.out_h() * params.c_out * params.b_size * kernel_layout.shape().elem_count() )

    );
    return Ok(());
}

pub fn queue_conv2d_transpose(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    dtype: crate::DType,
    params: &crate::conv::ParamsConvTranspose2D,
    input_layout: &crate::Layout,
    kernel_layout: &crate::Layout,
) -> crate::Result<()> {
    let input_stride = input_layout.stride();
    let kernel_stride = kernel_layout.stride();
    
    let (mut meta,  meta_offset) = get_meta(&dev, 24);
    meta.add(params.b_size);
    meta.add(params.c_in);
    meta.add(params.k_w);
    meta.add(params.k_h);
    meta.add(kernel_stride[3]); //kernel_x_stride
    meta.add(kernel_stride[2]); //kernel_y_stride
    meta.add(kernel_stride[0]); //kernel_c_stride
    meta.add(kernel_stride[1]); //kernel_b_stride
    meta.add(kernel_layout.start_offset());
    meta.add(params.i_w);   //size_in_x
    meta.add(params.i_h);   //size_in_y
    meta.add(params.out_w() * params.out_h() * params.c_out); //Stride_batch_out
    meta.add(params.out_w() * params.out_h()); //stride_c_out
    meta.add(params.out_w()); //stride_y_out
    meta.add(params.out_h()); //size_y_out

    meta.add(input_stride[0]); //stride_batch_input
    meta.add(input_stride[1]); //stride_c_in
    meta.add(input_stride[2]); //stride_y_in
    meta.add(input_stride[3]); //stride_x_in
    meta.add(params.padding);
    meta.add(params.stride);
    meta.add(params.dilation);
    meta.add(input_layout.start_offset());

    let pipeline = dev.get_pipeline(super::Shader::Conv2D(dtype), Pipelines::Conv2DTranspose)?;

    let bind_group = create_bind_group_input2(
        dev,
        pipeline.clone(),
        meta_offset,
        buffer_dest,
        buffer_input1,
        buffer_input2,
    );
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        ((params.out_w() - params.output_padding) as u32 + 7) / 8,
        ((params.out_h() - params.output_padding) as u32 + 7) / 8,
        params.c_out as u32,
        #[cfg(feature = "wgpu_debug")] crate::wgpu::device::QueueDebugInfo::new(&format!("conv2d_transpose, dtype:{:?}", dtype),params.out_w() * params.out_h() * params.c_out * params.b_size * kernel_layout.shape().elem_count()),
    );
    return Ok(());
}



pub fn queue_conv1d(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    dtype: crate::DType,
    params: &crate::conv::ParamsConv1D,
    input_layout: &crate::Layout,
    kernel_layout: &crate::Layout,
) -> crate::Result<()> {
    let input_stride = input_layout.stride();
    let kernel_stride = kernel_layout.stride();
    
    let (mut meta,  meta_offset) = get_meta(&dev, 18);

    meta.add(params.b_size);
    meta.add(params.c_in);
    meta.add(params.k_size);
    meta.add(kernel_stride[2]); //kernel_x_stride
    meta.add(kernel_stride[1]); //kernel_c_stride
    meta.add(kernel_stride[0]); //kernel_b_stride
    meta.add(kernel_layout.start_offset());
    meta.add(params.l_in);   //size_in_x
    meta.add(params.l_out() * params.c_out); //Stride_batch_out
    meta.add(params.l_out()); //stride_c_out
    meta.add(params.l_out()); //size_y_out

    meta.add(input_stride[0]); //stride_batch_input
    meta.add(input_stride[1]); //stride_c_in
    meta.add(input_stride[2]); //stride_x_in
    meta.add(params.padding);
    meta.add(params.stride);
    meta.add(params.dilation);
    meta.add(input_layout.start_offset());

    let pipeline = dev.get_pipeline(super::Shader::Conv2D(dtype), Pipelines::Conv1D)?;

    let bind_group = create_bind_group_input2(
        dev,
        pipeline.clone(),
        meta_offset,
        buffer_dest,
        buffer_input1,
        buffer_input2,
    );
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        (params.l_out() as u32 + 63) / 64,
        params.c_out as u32,
        1,
        #[cfg(feature = "wgpu_debug")] crate::wgpu::device::QueueDebugInfo::new(&format!("conv1d, dtype:{:?}", dtype), params.l_out() * params.c_out * params.b_size * kernel_layout.shape().elem_count()),
    );
    return Ok(());
}

pub fn queue_conv1d_transpose(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    dtype: crate::DType,
    params: &crate::conv::ParamsConvTranspose1D,
    input_layout: &crate::Layout,
    kernel_layout: &crate::Layout,
) -> crate::Result<()> {
    let input_stride = input_layout.stride();
    let kernel_stride = kernel_layout.stride();
    
    let (mut meta,  meta_offset) = get_meta(&dev, 18);
    meta.add(params.b_size);
    meta.add(params.c_in);
    meta.add(params.k_size);
    meta.add(kernel_stride[2]); //kernel_x_stride
    meta.add(kernel_stride[0]); //kernel_c_stride
    meta.add(kernel_stride[1]); //kernel_b_stride
    meta.add(kernel_layout.start_offset());
    meta.add(params.l_in);   //size_in_x
    meta.add(params.l_out() * params.c_out); //Stride_batch_out
    meta.add(params.l_out()); //stride_c_out
    meta.add(params.l_out()); //size_y_out

    meta.add(input_stride[0]); //stride_batch_input
    meta.add(input_stride[1]); //stride_c_in
    meta.add(input_stride[2]); //stride_x_in
    meta.add(params.padding);
    meta.add(params.stride);
    meta.add(params.dilation);
    meta.add(input_layout.start_offset());

    let pipeline = dev.get_pipeline(super::Shader::Conv2D(dtype), Pipelines::Conv1DTranspose)?;

    let bind_group = create_bind_group_input2(
        dev,
        pipeline.clone(),
        meta_offset,
        buffer_dest,
        buffer_input1,
        buffer_input2,
    );
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        ((params.l_out() - params.output_padding) as u32 + 63) / 64,
        params.c_out as u32,
        1u32,
        #[cfg(feature = "wgpu_debug")] 
        crate::wgpu::device::QueueDebugInfo::new(&format!("conv1d_transpose, dtype:{:?}", dtype), params.l_out() * params.c_out * params.b_size * kernel_layout.shape().elem_count()),
    );
    return Ok(());
}
