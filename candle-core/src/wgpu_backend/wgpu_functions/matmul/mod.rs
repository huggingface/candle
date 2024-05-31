use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, Layout, WgpuDevice};

use super::{create_bind_group_input2, enqueue_workgroups};



//(M X N) * (N X K)
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MetaInfoMatMul {
    b: u32, //Batches
    m: u32, //elements
    n: u32, //elements
    k: u32, //elements
    input1_stride_b: u32,
    input1_stride_m: u32,
    input1_stride_n: u32,
    input1_offset: u32,
    input2_stride_b: u32,
    input2_stride_n: u32,
    input2_stride_k: u32,
    input2_offset: u32,
}


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

    let meta = MetaInfoMatMul {
        b,
        m,
        n,
        k,
        input1_stride_n: *input1_stride.next().unwrap_or(&1) as u32,
        input1_stride_m: *input1_stride.next().unwrap_or(&1) as u32,
        input1_stride_b: *input1_stride.next().unwrap_or(&1) as u32,
        input1_offset: layout_input1.start_offset() as u32,

        input2_stride_k: *input2_stride.next().unwrap_or(&1) as u32,
        input2_stride_n: *input2_stride.next().unwrap_or(&1) as u32,
        input2_stride_b: *input2_stride.next().unwrap_or(&1) as u32,
        input2_offset: layout_input2.start_offset() as u32,
    };

    let pipeline = dev.get_pipeline(super::Shader::Matmul(dtype), Pipelines::MatmulBuffer)?;
  
    let bind_group = create_bind_group_input2(
        dev,
        pipeline.clone(),
        meta,
        buffer_dest,
        buffer_input1,
        buffer_input2,
    );
    enqueue_workgroups(
        dev,
        pipeline,
        bind_group,
        (k + 7) / 8,
        (m + 7) / 8,
        b,
        #[cfg(feature = "wgpu_debug")]&format!("matmul, dtype:{:?}", dtype),
    );
    return Ok(());
}
