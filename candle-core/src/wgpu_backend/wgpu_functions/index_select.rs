use super::*;
use crate::{wgpuError, Shape};

pub fn queue_index_select(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    input : WgpuTensor,
    index : WgpuTensor,
    dtype: crate::DType,
    index_dtype: crate::DType,
    dim: usize,
) -> crate::Result<()> {
    let index_length = index.layout().shape().elem_count();
    let length = (input.layout().shape().elem_count() / input.layout().shape().dims()[dim]) as u32;

    let mut new_shape = input.layout().shape().clone().into_dims();
    new_shape[dim] = index_length;
    let new_stride = Shape::from(new_shape.clone()).stride_contiguous();

    let output_stride_y = new_shape[(dim + 1)..].iter().fold(1, |prev, c| prev * *c) as u32; //Mul All Shapes after dim
    let input_stride_y = output_stride_y;
    let output_stride_x = new_stride[0..dim].iter().fold(1, |prev, c| prev * *c) as u32; //Mul all New Strides left of dim
    let input_stride_x = input.layout().stride()[0..dim]
        .iter()
        .fold(1, |prev, c| prev * *c) as u32; //Mul Strides Left of dim

    let mut meta = get_meta(dev);

    meta.add(input_stride_x);
    meta.add(input_stride_y);
    meta.add(output_stride_x);
    meta.add(output_stride_y);
    meta.add(length);
    meta.add_layout1(input.layout());
    meta.add_layout2(index.layout());

    let pipeline = match index_dtype {
        crate::DType::U32 => Pipelines::IndexSelect(get_dtype(dtype)?, candle_wgpu_kernels::index_select::Functions::IndexSelectU32),
        crate::DType::I64 => Pipelines::IndexSelecti64(get_dtype(dtype)?, candle_wgpu_kernels::index_selecti64::Functions::IndexSelectI64),
        _ => wgpuError!(format!("dtype: {:?} is not supported for indexing in index select", index_dtype)),
    };
    let pipeline = meta.get_pipeline(pipeline);

    let bind_group = create_bind_group_input2_with_alignment(
        buffer_dest,
        index.buffer(),
        input.buffer(),
        BindgroupAlignmentLayout::Bindgroup2(dtype.into(), index_dtype.into(), dtype.into()),
    );
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        (length + 7) / 8,
        ((index_length + 7) / 8) as u32,
        1,
        length as usize * index_length,
    );
    Ok(())
}
