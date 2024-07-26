use candle_wgpu_kernels::index_select::Functions;

use crate::Shape;
use super::*;

pub fn queue_index_select(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    buffer_index: BufferReferenceId,
    dtype: crate::DType,
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

    let mut meta = get_meta(&dev);

    meta.add(input_stride_x);
    meta.add(input_stride_y);
    meta.add(output_stride_x);
    meta.add(output_stride_y);
    meta.add(length);
    meta.add_layout(&lay_input);
    meta.add_layout(&lay_index);

    let pipeline = get_pipeline(Pipelines::IndexSelect(get_dtype(dtype)?, Functions::IndexSelect));

    let bind_group = create_bind_group_input2( buffer_dest, buffer_input, buffer_index);
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        (length + 7) / 8,
        ((index_length + 7) / 8) as u32,
        1,
        length as usize * index_length
    );
    return Ok(());
}
