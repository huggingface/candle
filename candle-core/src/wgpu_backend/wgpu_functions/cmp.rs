use super::*;
use crate::Layout;
use candle_wgpu_kernels::cmp::Functions;

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
    buffer_dest: BufferReferenceId,
    buffer_input1: BufferReferenceId,
    buffer_input2: BufferReferenceId,
    op: CmpOperation,
    dtype: crate::DType,
    layout_input1: &Layout,
    layout_input2: &Layout,
) -> crate::Result<()> {
    let mut meta = get_meta(&dev);
    meta.add(op as u32);
    meta.add_layout1(&layout_input1);
    meta.add_layout2(&layout_input2);

    let pipeline = meta.get_pipeline(Pipelines::Cmp(
        get_dtype(dtype)?,
        Functions::CmpBufferFromBuffer,
    ));

    let bind_group = create_bind_group_input2(buffer_dest, buffer_input1, buffer_input2, dtype.into());
    enqueue(
        meta,
        pipeline,
        bind_group,
        ((layout_input1.shape().elem_count() + 3) / 4) as u32,
        layout_input1.shape().elem_count(),
    );
    return Ok(());
}
