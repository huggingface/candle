use super::*;
use candle_wgpu_kernels::gather::Functions;

pub fn queue_gather(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    input: WgpuTensor,
    index: WgpuTensor,
    dtype: crate::DType,
    dim: usize,
) -> crate::Result<()> {
    let mut queue = dev.get_queue();

    queue.add(dim);
    queue.add_layout1_non_contiguous(input.layout());
    queue.add_layout2_non_contiguous(index.layout());

    let pipeline = queue.get_pipeline(Pipelines::Gather(dev.get_dtype(dtype)?, Functions::Gather));

    let bind_group =
        dev.create_bind_group_input2(buffer_dest, input.buffer(), index.buffer(), dtype.into());
    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        (index.layout().shape().elem_count() as u32).div_ceil(64),
        1,
        1,
        index.layout().shape().elem_count(),
    );
    Ok(())
}

pub fn queue_scatter_add_inplace(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    index: WgpuTensor,
    src: WgpuTensor,
    dtype: crate::DType,
    lay_input: &crate::Layout,
    dim: usize,
) -> crate::Result<()> {
    let mut queue = dev.get_queue();

    let selected_index_length = index.layout().shape().dims()[dim];

    queue.add(dim);
    queue.add_layout1_non_contiguous(lay_input);
    queue.add_layout2_non_contiguous(index.layout());
    queue.add_layout3_non_contiguous(src.layout());

    let pipeline = queue.get_pipeline(Pipelines::Gather(
        dev.get_dtype(dtype)?,
        Functions::ScatterAddInplace,
    ));

    let bind_group =
        dev.create_bind_group_input2(buffer_dest, index.buffer(), src.buffer(), dtype.into());
    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        ((index.layout().shape().elem_count() / selected_index_length) as u32).div_ceil(64),
        1,
        1,
        index.layout().shape().elem_count(),
    );
    Ok(())
}

pub fn queue_scatter_set_inplace(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    index: WgpuTensor,
    src: WgpuTensor,
    dtype: crate::DType,
    lay_input: &crate::Layout,
    dim: usize,
) -> crate::Result<()> {
    let mut queue = dev.get_queue();

    let selected_index_length = index.layout().shape().dims()[dim];

    queue.add(dim);
    queue.add_layout1_non_contiguous(lay_input);
    queue.add_layout2_non_contiguous(index.layout());
    queue.add_layout3_non_contiguous(src.layout());

    let pipeline = queue.get_pipeline(Pipelines::Gather(
        dev.get_dtype(dtype)?,
        Functions::ScatterSetInplace,
    ));

    let bind_group =
        dev.create_bind_group_input2(buffer_dest, index.buffer(), src.buffer(), dtype.into());
    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        ((index.layout().shape().elem_count() / selected_index_length) as u32).div_ceil(64),
        1,
        1,
        index.layout().shape().elem_count(),
    );
    Ok(())
}

pub fn queue_index_add_inplace(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    index: WgpuTensor,
    src: WgpuTensor,
    dtype: crate::DType,
    lay_input: &crate::Layout,
    dim: usize,
) -> crate::Result<()> {
    let mut queue = dev.get_queue();

    let selected_index_length = index.layout().shape().elem_count();

    queue.add(dim);
    queue.add_layout1_non_contiguous(lay_input);
    queue.add_layout2_non_contiguous(index.layout());
    queue.add_layout3_non_contiguous(src.layout());

    let pipeline = queue.get_pipeline(Pipelines::Gather(
        dev.get_dtype(dtype)?,
        Functions::IndexAddInplace,
    ));

    let bind_group =
        dev.create_bind_group_input2(buffer_dest, index.buffer(), src.buffer(), dtype.into());
    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        ((lay_input.shape().elem_count() / selected_index_length) as u32).div_ceil(64),
        1,
        1,
        lay_input.shape().elem_count(),
    );
    Ok(())
}
