use std::sync::Arc;


use crate::{wgpu::{cache::BufferReference, device::Pipelines}, WgpuDevice};

use super::{create_bind_group_input2, enqueue_workgroups, get_meta};



pub fn queue_gather(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input: Arc<BufferReference>,
    buffer_index: Arc<BufferReference>,
    input_dtype: crate::DType,
    lay_input: &crate::Layout,
    lay_index: &crate::Layout,
    dim: usize,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);

    meta.add(dim);
    meta.add_layout(&lay_input);
    meta.add_layout(&lay_index);

    let pipeline = dev.get_pipeline(super::Shader::Gather(input_dtype), Pipelines::Gather)?;

    let bind_group =
        create_bind_group_input2( buffer_dest, buffer_input, buffer_index);
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        (lay_index.shape().elem_count() as u32 + 63) / 64,
        1,
        1,
        #[cfg(feature = "wgpu_debug")] 
        crate::wgpu::device::QueueDebugInfo::new(&format!("gather : dtype{:?}", input_dtype),lay_index.shape().elem_count()),
    );
    return Ok(());
}



pub fn queue_scatter_add_inplace(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_index: Arc<BufferReference>,
    buffer_src: Arc<BufferReference>,
    input_dtype: crate::DType,
    lay_input: &crate::Layout,
    lay_index: &crate::Layout,
    lay_src: &crate::Layout,
    dim: usize,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);

    let selected_index_length = lay_index.shape().dims()[dim];

    meta.add(dim);
    meta.add_layout(&lay_input);
    meta.add_layout(&lay_index);
    meta.add_layout(&lay_src);

    let pipeline = dev.get_pipeline(super::Shader::Gather(input_dtype), Pipelines::ScatterAddInplace)?;

    let bind_group =
        create_bind_group_input2( buffer_dest, buffer_index, buffer_src);
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        ((lay_index.shape().elem_count() / selected_index_length) as u32 + 63) / 64,
        1,
        1,
        #[cfg(feature = "wgpu_debug")] 
        crate::wgpu::device::QueueDebugInfo::new(&format!("scatter_add : dtype{:?}", input_dtype), lay_index.shape().elem_count()),
    );
    return Ok(());
}

pub fn queue_index_add_inplace(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_index: Arc<BufferReference>,
    buffer_src: Arc<BufferReference>,
    input_dtype: crate::DType,
    lay_input: &crate::Layout,
    lay_index: &crate::Layout,
    lay_src: &crate::Layout,
    dim: usize,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);

    let selected_index_length = lay_index.shape().elem_count();

    meta.add(dim);
    meta.add_layout(&lay_input);
    meta.add_layout(&lay_index);
    meta.add_layout(&lay_src);

    let pipeline = dev.get_pipeline(super::Shader::Gather(input_dtype), Pipelines::IndexAddInplace)?;

    let bind_group =
        create_bind_group_input2( buffer_dest, buffer_index, buffer_src);
    enqueue_workgroups(
        meta,
        pipeline,
        bind_group,
        ((lay_input.shape().elem_count() / selected_index_length) as u32 + 63) / 64,
        1,
        1,
        #[cfg(feature = "wgpu_debug")] 
        crate::wgpu::device::QueueDebugInfo::new(&format!("scatter_add : dtype{:?}", input_dtype), lay_input.shape().elem_count()),
    );
    return Ok(());
}
