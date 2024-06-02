use wgpu::Buffer;

use crate::{wgpu::device::Pipelines, Layout, WgpuDevice};

use super::{create_bind_group_input2, enqueue, MatrixLayout, MyArray};

// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// #[repr(C)]
// struct MetaBinary {
//     input1_layout: MatrixLayout,
//     input2_layout: MatrixLayout,
//     operation: u32,
// }


#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub enum CmpOperation {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}


pub fn queue_cmp_buffer_from_buffer(
    dev: &WgpuDevice,
    buffer_dest: &Buffer,
    buffer_input1: &Buffer,
    buffer_input2: &Buffer,
    op: CmpOperation,
    dtype: crate::DType,
    layout_input1: &Layout,
    layout_input2: &Layout,
) -> crate::Result<()> {

    let mut meta = MyArray::new(10);
    meta.add(op as u32);
    meta.add_layout(&layout_input1);
    meta.add_layout(&layout_input2);
    // let meta = MetaBinary {
    //     operation: op as u32,
    //     input1_layout: MatrixLayout::from_layout(&layout_input1),
    //     input2_layout: MatrixLayout::from_layout(&layout_input2),
    // };

    let pipeline = dev.get_pipeline(super::Shader::Cmp(dtype), Pipelines::CmpFromBuffer)?;

    let bind_group = create_bind_group_input2(
        dev,
        pipeline.clone(),
        &meta.0,
        buffer_dest,
        buffer_input1,
        buffer_input2,
    );
    enqueue(
        dev,
        pipeline,
        bind_group,
        ((layout_input1.shape().elem_count() + 3) / 4) as u32,
        #[cfg(feature = "wgpu_debug")] &format!("cmp op:{:?}, dtype:{:?}", op, dtype),
    );
    return Ok(());
}
