use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, Layout, WgpuDevice};

use super::{create_bind_group_input2, enqueue_workgroups, get_meta};

pub fn queue_matmul_buffer(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    buffer_input2: &Buffer,
    b: u32,
    m: u32,
    n: u32,
    k: u32,
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    let mut input1_stride = layout_input1.stride().iter().rev();
    let mut input2_stride = layout_input2.stride().iter().rev();

    let (mut meta,  meta_offset) = get_meta(&dev, 12);
    meta.add(b);
    meta.add(m);
    meta.add(n);
    meta.add(k);

    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_n
    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_m
    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_b
    meta.add(layout_input1.start_offset()); //input1_offset

    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_k
    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_n
    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_b
    meta.add(layout_input2.start_offset()); //input2_offset

    let pipeline = dev.get_pipeline(super::Shader::Matmul(dtype), Pipelines::MatmulBuffer)?;
  
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
        (k  + 15) / 16,
        (m  + 15) / 16,
        b,
        #[cfg(feature = "wgpu_debug")]
        crate::wgpu::device::QueueDebugInfo::new(&format!("matmul, dtype:{:?}", dtype), k * m * n),
    );
    return Ok(());
}


//shader3
pub fn queue_matmul_buffer3(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    buffer_input2: &Buffer,
    b: u32,
    m: u32,
    n: u32,
    k: u32,
    layout_input1: &Layout,
    layout_input2: &Layout,
    dtype: crate::DType,
) -> crate::Result<()> {
    let mut input1_stride = layout_input1.stride().iter().rev();
    let mut input2_stride = layout_input2.stride().iter().rev();

    let (mut meta,  meta_offset) = get_meta(&dev, 12);
    meta.add(b);
    meta.add(m);
    meta.add(n);
    meta.add(k);

    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_n
    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_m
    meta.add(*input1_stride.next().unwrap_or(&1)); //input1_stride_b
    meta.add(layout_input1.start_offset()); //input1_offset

    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_k
    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_n
    meta.add(*input2_stride.next().unwrap_or(&1)); //input2_stride_b
    meta.add(layout_input2.start_offset()); //input2_offset

    let pipeline = dev.get_pipeline(super::Shader::Matmul(dtype), Pipelines::Matmul3Buffer)?;
  
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
        (k+ 7) / 8,
        (m + 7) / 8,
        b,
        #[cfg(feature = "wgpu_debug")]
        crate::wgpu::device::QueueDebugInfo::new(&format!("matmul, dtype:{:?}", dtype), k * m * n),
    );
    return Ok(());
}
