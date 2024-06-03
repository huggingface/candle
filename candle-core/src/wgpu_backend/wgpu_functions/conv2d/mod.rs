use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, WgpuDevice};

use super::{create_bind_group_input2, enqueue_workgroups, get_meta};



// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// #[repr(C)]
// struct MetaConv2d {
//     b: u32,    //batch_count ("normal" matmul = 1)
//     c_in: u32, //Output Channel, we are using workgroups for all c_out, x, y pairs
//     kernel_x: u32,
//     kernel_y: u32,
//     kernel_x_stride: u32,
//     kernel_y_stride: u32,
//     kernel_c_stride: u32,
//     kernel_b_stride: u32,
//     kernel_offset: u32,
//     size_in_x: u32,
//     size_in_y: u32,
//     stride_batch_out: u32,
//     stride_c_out: u32,
//     stride_y_out: u32,
//     size_y_out: u32,

//     stride_batch_input: u32,
//     stride_c_in: u32,
//     stride_y_in: u32,
//     stride_x_in: u32,
//     padding: u32,
//     output_padding : u32,
//     stride_conv: u32,
//     dialation_conv: u32,
//     offset_input: u32,
// }


pub fn queue_conv2d(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    buffer_input2: &Buffer,
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
    meta.add(0u32);   //output padding
    meta.add(params.stride);
    meta.add(params.dilation);
    meta.add(input_layout.start_offset());

    // let meta = MetaConv2d {
    //     b: params.b_size as u32,
    //     c_in: params.c_in as u32,
    //     kernel_x: params.k_w as u32,
    //     kernel_y: params.k_h as u32,
    //     size_in_x: params.i_w as u32,
    //     size_in_y: params.i_h as u32,
    //     stride_batch_out: (params.out_w() * params.out_h() * params.c_out) as u32,
    //     stride_c_out: (params.out_w() * params.out_h()) as u32,
    //     stride_y_out: params.out_w() as u32,
    //     size_y_out: params.out_h() as u32,

    //     stride_batch_input: input_stride[0] as u32,
    //     stride_c_in: input_stride[1] as u32,
    //     stride_y_in: input_stride[2] as u32,
    //     stride_x_in: input_stride[3] as u32,

    //     kernel_b_stride: kernel_stride[0] as u32,
    //     kernel_c_stride: kernel_stride[1] as u32,
    //     kernel_y_stride: kernel_stride[2] as u32,
    //     kernel_x_stride: kernel_stride[3] as u32,
    //     kernel_offset: kernel_layout.start_offset() as u32,

    //     // stride_batch_input: (params.i_h * params.i_w * params.c_in) as u32,
    //     // stride_c_in: (params.i_w * params.i_h) as u32,
    //     // stride_y_in: params.i_w  as u32,
    //     padding: params.padding as u32,
    //     output_padding : 0,
    //     stride_conv: params.stride as u32,
    //     dialation_conv: params.dilation as u32,
    //     offset_input: input_layout.start_offset() as u32,
    // };

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
        dev,
        pipeline,
        bind_group,
        (params.out_w() as u32 + 7) / 8,
        (params.out_h() as u32 + 7) / 8,
        params.c_out as u32,
        #[cfg(feature = "wgpu_debug")] &format!("conv2d, dtype:{:?}", dtype),
    );
    return Ok(());
}

pub fn queue_conv2d_transpose(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    buffer_input2: &Buffer,
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
    meta.add(params.output_padding);   //output padding
    meta.add(params.stride);
    meta.add(params.dilation);
    meta.add(input_layout.start_offset());

    

    
    // let meta = MetaConv2d {
    //     b: params.b_size as u32,
    //     c_in: params.c_in as u32,
    //     kernel_x: params.k_w as u32,
    //     kernel_y: params.k_h as u32,
    //     size_in_x: params.i_w as u32,
    //     size_in_y: params.i_h as u32,
    //     stride_batch_out: (params.out_w() * params.out_h() * params.c_out) as u32,
    //     stride_c_out: (params.out_w() * params.out_h()) as u32,
    //     stride_y_out: params.out_w() as u32,
    //     size_y_out: params.out_h() as u32,

    //     //stride_batch_input: (params.i_h * params.i_w * params.c_in) as u32,
    //     //stride_c_in: (params.i_w * params.i_h) as u32,
    //     //stride_y_in: params.i_w  as u32,
    //     stride_batch_input: input_stride[0] as u32,
    //     stride_c_in: input_stride[1] as u32,
    //     stride_y_in: input_stride[2] as u32,
    //     stride_x_in: input_stride[3] as u32,

    //     kernel_b_stride: kernel_stride[1] as u32,
    //     kernel_c_stride: kernel_stride[0] as u32,
    //     kernel_y_stride: kernel_stride[2] as u32,
    //     kernel_x_stride: kernel_stride[3] as u32,

    //     //kernel_c_stride : (params.k_w * params.k_h) as u32,
    //     //kernel_y_stride : params.k_w as u32,
    //     //kernel_b_stride : (params.k_w * params.k_h * params.c_in) as u32,
    //     //kernel_x_stride : 1,
    //     kernel_offset: kernel_layout.start_offset() as u32,

    //     padding: params.padding as u32,
    //     output_padding: params.output_padding as u32,
    //     stride_conv: params.stride as u32,
    //     dialation_conv: params.dilation as u32,
    //     offset_input: input_layout.start_offset() as u32,
    // };

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
        dev,
        pipeline,
        bind_group,
        ((params.out_w() - params.output_padding) as u32 + 7) / 8,
        ((params.out_h() - params.output_padding) as u32 + 7) / 8,
        params.c_out as u32,
        #[cfg(feature = "wgpu_debug")] &format!("conv2d_transpose, dtype:{:?}", dtype),
    );
    return Ok(());
}
