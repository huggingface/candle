use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, WgpuDevice};

use super::{create_bind_group_input1, enqueue_workgroups, get_meta};

pub fn queue_upsample1d(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    layout: &crate::Layout,
    dtype: crate::DType,
    target_size: usize,
) -> crate::Result<()> {
    let (b, c, l) = layout.shape().dims3()?;

    let strides = layout.stride();

    let (mut meta,  meta_offset) = get_meta(&dev, 9);

    meta.add(target_size);
    meta.add(b);
    meta.add(c);
    meta.add(l);
    meta.add(layout.start_offset());

    meta.add(strides[0]);
    meta.add(strides[1]);
    meta.add(strides[2]);
    
    meta.add(c * target_size);
    meta.add(target_size);

    let pipeline = dev.get_pipeline(super::Shader::Upsample(dtype), Pipelines::Upsample1d)?;

    let bind_group = create_bind_group_input1(
        dev,
        pipeline.clone(),
        meta_offset,
        buffer_dest,
        buffer_input1,
    );
    enqueue_workgroups(
        dev,
        pipeline,
        bind_group,
        (target_size as u32 + 63) / 63,
        c as u32,
        b as u32,
        #[cfg(feature = "wgpu_debug")] &format!("upsample1d, dtype:{:?}", dtype),
    );
    return Ok(());
}


pub fn queue_upsample2d(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    layout: &crate::Layout,
    dtype: crate::DType,
    target_size: (usize, usize),
) -> crate::Result<()> {
    let (b, c, h, w) = layout.shape().dims4()?;

    let strides = layout.stride();

    let (mut meta,  meta_offset) = get_meta(&dev, 14);

    meta.add(target_size.0);
    meta.add(target_size.1);
    meta.add(b);
    meta.add(c);
    meta.add(h);
    meta.add(w);
    meta.add(layout.start_offset());

    meta.add(strides[0]);
    meta.add(strides[1]);
    meta.add(strides[2]);
    meta.add(strides[3]);
    
    meta.add(c * target_size.0 * target_size.1);
    meta.add(target_size.0 * target_size.1);
    meta.add(target_size.1);

    let pipeline = dev.get_pipeline(super::Shader::Upsample(dtype), Pipelines::Upsample2d)?;

    let bind_group = create_bind_group_input1(
        dev,
        pipeline.clone(),
        meta_offset,
        buffer_dest,
        buffer_input1,
    );
    enqueue_workgroups(
        dev,
        pipeline,
        bind_group,
        (target_size.1 as u32 + 7) / 8,
        (target_size.0 as u32 + 7) / 8,
        c as u32,
        #[cfg(feature = "wgpu_debug")] &format!("upsample2d, dtype:{:?}", dtype),
    );
    return Ok(());
}

