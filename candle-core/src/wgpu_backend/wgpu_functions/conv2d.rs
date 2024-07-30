use candle_wgpu_kernels::conv2d::Functions;
use candle_wgpu_kernels::conv1d::Functions as Functions1d;

use super::*;

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
    
    let mut meta = get_meta(&dev);

    let const_vec = vec![
        kernel_stride[3],//kernel_x_stride
        //kernel_stride[2],//kernel_y_stride
        //input_stride[2], //stride_y_in
        input_stride[3], //stride_x_in
        params.padding,
        params.stride,
        params.dilation,
        input_layout.start_offset(),
        params.k_w, 
        params.k_h,
        params.b_size,
        params.c_in];


    meta.add(params.b_size);
    meta.add(params.c_in);
    meta.add(params.k_w);
    meta.add(params.k_h);
    //meta.add(kernel_stride[3]); //kernel_x_stride
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
    //meta.add(input_stride[3]); //stride_x_in
    //meta.add(params.padding);
    //meta.add(params.stride);
    //meta.add(params.dilation);
    //meta.add(input_layout.start_offset());
    let pipeline = meta.get_pipeline_const(Pipelines::Conv2d(get_dtype(dtype)?, Functions::Conv2d), const_vec);

    let bind_group = create_bind_group_input2(
        buffer_dest,
        buffer_input1,
        buffer_input2,
    );
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        (params.out_w() as u32 + 15) / 16,
        (params.out_h() as u32 + 15) / 16,
        params.c_out as u32,
        params.out_w() * params.out_h() * params.c_out * params.b_size * kernel_layout.shape().elem_count(),
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
    
    let mut meta = get_meta(&dev);
    
    let const_vec = vec![
        kernel_stride[3],//kernel_x_stride
        //kernel_stride[2],//kernel_y_stride
        //input_stride[2], //stride_y_in
        input_stride[3], //stride_x_in
        params.padding,
        params.stride,
        params.dilation,
        input_layout.start_offset(),
        params.k_w,
        params.k_h,
        params.b_size,
        params.c_in
        ];
    
    meta.add(params.b_size);
    meta.add(params.c_in);
    meta.add(params.k_w);
    meta.add(params.k_h);
    //meta.add(kernel_stride[3]); //kernel_x_stride
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
    //meta.add(input_stride[3]); //stride_x_in
    //meta.add(params.padding);
    //meta.add(params.stride);
    //meta.add(params.dilation);
    //meta.add(input_layout.start_offset());
    let pipeline = meta.get_pipeline_const(Pipelines::Conv2d(get_dtype(dtype)?, Functions::Conv2dTranspose), const_vec);
    let bind_group = create_bind_group_input2(
        buffer_dest,
        buffer_input1,
        buffer_input2,
    );
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        ((params.out_w() - params.output_padding) as u32 + 15) / 16,
        ((params.out_h() - params.output_padding) as u32 + 15) / 16,
        params.c_out as u32,
        params.out_w() * params.out_h() * params.c_out * params.b_size * kernel_layout.shape().elem_count(),
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

    let const_vec = vec![
        kernel_stride[2],//kernel_x_stride
        input_stride[2], //stride_x_in
        params.padding,
        params.stride,
        params.dilation,
        input_layout.start_offset(),
        params.k_size,
        params.b_size,
        params.c_in
        ];
    let mut meta = get_meta(&dev);

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

    let pipeline = meta.get_pipeline_const(Pipelines::Conv1d(get_dtype(dtype)?, Functions1d::Conv1d), const_vec);

    let bind_group = create_bind_group_input2(
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
        params.l_out() * params.c_out * params.b_size * kernel_layout.shape().elem_count(),
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
    
    let const_vec = vec![
        kernel_stride[2],//kernel_x_stride
        input_stride[2], //stride_x_in
        params.padding,
        params.stride,
        params.dilation,
        input_layout.start_offset(),
        params.k_size,
        params.b_size,
        params.c_in
        ];
    let mut meta = get_meta(&dev);
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

    let pipeline = meta.get_pipeline_const(Pipelines::Conv1d(get_dtype(dtype)?, Functions1d::Conv1dTranspose), const_vec);
    let bind_group = create_bind_group_input2(
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
        params.l_out() * params.c_out * params.b_size * kernel_layout.shape().elem_count(),
    );
    return Ok(());
}
