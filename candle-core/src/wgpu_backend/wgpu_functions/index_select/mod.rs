use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, Shape, WgpuDevice};

use super::{create_bind_group_input2, enqueue_workgroups_indirect, MatrixLayout, MyArray};



// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// #[repr(C)]
// struct MetaIndexSelect {
//     input1_layout: MatrixLayout,
//     input2_layout: MatrixLayout,
//     input_stride_x: u32,  //x specifys for values of one dim
//     input_stride_y: u32,  //y specifys per value of the index
//     output_stride_x: u32, //x specifys for values of one dim
//     output_stride_y: u32, //y specifys per value of the index
//     length: u32,
// }


pub fn queue_index_select(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input: &Buffer,
    buffer_index: &Buffer,
    input_dtype: crate::DType,
    lay_input: &crate::Layout,
    lay_index: &crate::Layout,
    dim: usize,
) -> crate::Result<()> {
    let index_length = lay_index.shape().elem_count();
    let length = (lay_input.shape().elem_count() / lay_input.shape().dims()[dim]) as u32;

    let mut new_shape = lay_input.shape().clone().into_dims();
    new_shape[dim] = index_length;
    let new_stride = Shape::from(new_shape.clone()).stride_contiguous();

    let output_stride_y = new_shape[(dim + 1)..].iter().fold(1, |prev, c| prev * *c) as u32; //Mul All Shapes after dim
    let input_stride_y = output_stride_y as u32;
    let output_stride_x = new_stride[0..dim].iter().fold(1, |prev, c| prev * *c) as u32; //Mul all New Strides left of dim
    let input_stride_x = lay_input.stride()[0..dim]
        .iter()
        .fold(1, |prev, c| prev * *c) as u32; //Mul Strides Left of dim

    let mut meta = MyArray::new(15);
    meta.add(input_stride_x);
    meta.add(input_stride_y);
    meta.add(output_stride_x);
    meta.add(output_stride_y);
    meta.add(length);
    meta.add_layout(&lay_input);
    meta.add_layout(&lay_index);

    
    
    // let meta = MetaIndexSelect {
    //     input1_layout: MatrixLayout::from_layout(&lay_input),
    //     input2_layout: MatrixLayout::from_layout(&lay_index),
    //     length,
    //     input_stride_x,
    //     input_stride_y,
    //     output_stride_x,
    //     output_stride_y,
    // };

    let pipeline = dev.get_pipeline(super::Shader::IndexSelect(input_dtype), Pipelines::IndexSelect)?;

    let bind_group =
        create_bind_group_input2(dev, pipeline.clone(), &meta.0, buffer_dest, buffer_input, buffer_index);
    enqueue_workgroups_indirect(
        dev,
        pipeline,
        bind_group,
        (length + 7) / 8,
        ((lay_index.shape().elem_count() + 7) / 8) as u32,
        1,
        #[cfg(feature = "wgpu_debug")] &format!("index_select : dtype{:?}", input_dtype),
    );
    return Ok(());
}
