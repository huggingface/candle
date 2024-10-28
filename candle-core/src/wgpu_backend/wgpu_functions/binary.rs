use candle_wgpu_kernels::binary::Functions;

use super::*;

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub enum BinaryOperation {
    SetY = 0,
    Add = 1,
    Mult = 2,
    Minus = 3,
    Div = 4,
    Max = 5,
    Min = 6,
    Pow = 7,
}
pub fn queue_binary_buffer_from_buffer(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    input1: WgpuTensor,
    input2: WgpuTensor,
    op: BinaryOperation,
    dtype: crate::DType,
) -> crate::Result<()> {
    let mut meta = get_meta(dev);
    let pipeline = if input1.layout().is_contiguous() && input2.layout().is_contiguous() {
        let const_vec = vec![
            op as usize,
            (input1.layout().start_offset() == 0) as usize,
            (input2.layout().start_offset() == 0) as usize,
        ];

        meta.add(input1.layout().shape().elem_count()); //input1_length
        meta.add(input1.layout().start_offset());
        meta.add(input2.layout().start_offset());

        let inplaceable = OpIsInplaceable {
            input1_inplaceable: input1.layout().start_offset() == 0,
            input2_inplaceable: input2.layout().start_offset() == 0,
        };

        if input1.layout().shape().elem_count() > 65535 * 64 {
            meta.add_const(candle_wgpu_kernels::Constants::UseZ, true);
        }

        meta.get_pipeline_const_inplace(
            Pipelines::Binary(
                get_dtype(dtype)?,
                Functions::BinaryBufferFromBufferContiguousBoth,
            ),
            const_vec,
            inplaceable,
        )
    } else {
        let const_vec = vec![op as usize];
        meta.add_layout1(input1.layout());
        meta.add_layout2(input2.layout());

        if input1.layout().shape().elem_count() > 65535 * 64 {
            meta.add_const(candle_wgpu_kernels::Constants::UseZ, true);
        }

        meta.get_pipeline_const(
            Pipelines::Binary(get_dtype(dtype)?, Functions::BinaryBufferFromBuffer),
            const_vec,
        )
    };

    let bind_group =
        create_bind_group_input2(buffer_dest, input1.buffer(), input2.buffer(), dtype.into());

    enqueue_big_extra(
        meta,
        pipeline,
        bind_group,
        input1.layout().shape().elem_count() as u32,
        #[cfg(feature = "wgpu_debug")]
        Some(format!("OP: {:?}, layout: {:?}", op, lay1)),
    );
    Ok(())
}
