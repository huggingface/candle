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
    input1: WgpuTensor,
    input2: WgpuTensor,
    op: CmpOperation,
    dtype: crate::DType,
) -> crate::Result<()> {
    let mut queue = dev.get_queue();
    queue.add(op as u32);
    queue.add_layout1(input1.layout());
    queue.add_layout2(input2.layout());

    let pipeline = queue.get_pipeline(Pipelines::Cmp(
        dev.get_dtype(dtype)?,
        Functions::CmpBufferFromBuffer,
    ));

    let bind_group =
        dev.create_bind_group_input2(buffer_dest, input1.buffer(), input2.buffer(), dtype.into());
    queue.enqueue_64(
        pipeline,
        bind_group,
        input1.layout().shape().elem_count().div_ceil(4) as u32,
        input1.layout().shape().elem_count(),
    );
    Ok(())
}
