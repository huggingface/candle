use candle_wgpu_kernels::gather::Functions;
use super::*;

pub fn queue_gather(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_input: BufferReferenceId,
    buffer_index: BufferReferenceId,
    dtype: crate::DType,
    lay_input: &crate::Layout,
    lay_index: &crate::Layout,
    dim: usize,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);

    meta.add(dim);
    meta.add_layout1_non_contiguous(&lay_input);
    meta.add_layout2_non_contiguous(&lay_index);

    let pipeline = meta.get_pipeline(Pipelines::Gather(get_dtype(dtype)?, Functions::Gather));

    let bind_group =
        create_bind_group_input2( buffer_dest, buffer_input, buffer_index);
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        (lay_index.shape().elem_count() as u32 + 63) / 64,
        1,
        1,
        lay_index.shape().elem_count()
    );
    return Ok(());
}



pub fn queue_scatter_add_inplace(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_index: BufferReferenceId,
    buffer_src: BufferReferenceId,
    dtype: crate::DType,
    lay_input: &crate::Layout,
    lay_index: &crate::Layout,
    lay_src: &crate::Layout,
    dim: usize,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);

    let selected_index_length = lay_index.shape().dims()[dim];

    meta.add(dim);
    meta.add_layout1_non_contiguous(&lay_input);
    meta.add_layout2_non_contiguous(&lay_index);
    meta.add_layout3_non_contiguous(&lay_src);

    let pipeline = meta.get_pipeline(Pipelines::Gather(get_dtype(dtype)?, Functions::ScatterAddInplace));

    let bind_group =
        create_bind_group_input2( buffer_dest, buffer_index, buffer_src);
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        ((lay_index.shape().elem_count() / selected_index_length) as u32 + 63) / 64,
        1,
        1,
        lay_index.shape().elem_count(),
    );
    return Ok(());
}

pub fn queue_index_add_inplace(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_index: BufferReferenceId,
    buffer_src: BufferReferenceId,
    dtype: crate::DType,
    lay_input: &crate::Layout,
    lay_index: &crate::Layout,
    lay_src: &crate::Layout,
    dim: usize,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);

    let selected_index_length = lay_index.shape().elem_count();

    meta.add(dim);
    meta.add_layout1_non_contiguous(&lay_input);
    meta.add_layout2_non_contiguous(&lay_index);
    meta.add_layout3_non_contiguous(&lay_src);

    let pipeline = meta.get_pipeline(Pipelines::Gather(get_dtype(dtype)?, Functions::IndexAddInplace));

    let bind_group =
        create_bind_group_input2( buffer_dest, buffer_index, buffer_src);
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        ((lay_input.shape().elem_count() / selected_index_length) as u32 + 63) / 64,
        1,
        1,
        lay_input.shape().elem_count(),
    );
    return Ok(());
}
