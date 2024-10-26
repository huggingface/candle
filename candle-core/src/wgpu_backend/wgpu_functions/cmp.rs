use super::*;
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
    input1 : WgpuTensor,
    input2 : WgpuTensor,
    op: CmpOperation,
    dtype: crate::DType,
) -> crate::Result<()> {
    let mut meta = get_meta(dev);
    meta.add(op as u32);
    meta.add_layout1(input1.layout());
    meta.add_layout2(input2.layout());

    let pipeline = meta.get_pipeline(Pipelines::Cmp(
        get_dtype(dtype)?,
        Functions::CmpBufferFromBuffer,
    ));

    let bind_group = create_bind_group_input2(buffer_dest, input1.buffer(), input2.buffer(), dtype.into());
    enqueue(
        meta,
        pipeline,
        bind_group,
        ((input1.layout().shape().elem_count() + 3) / 4) as u32,
        input1.layout().shape().elem_count(),
    );
    Ok(())
}
