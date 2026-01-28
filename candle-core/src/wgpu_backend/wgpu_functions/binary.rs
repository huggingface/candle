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
    let layout1 = normalize_layout(input1.layout());
    let layout2 = normalize_layout(input2.layout());
    let mut queue = dev.get_queue();
    let pipeline = if layout1.is_contiguous() && layout2.is_contiguous() {
        let const_vec = vec![
            op as usize,
            (layout1.start_offset() == 0) as usize,
            (layout2.start_offset() == 0) as usize,
        ];

        queue.add(layout1.shape().elem_count()); //input1_length
        queue.add(layout1.start_offset());
        queue.add(layout2.start_offset());

        let inplaceable = OpIsInplaceable {
            input1_inplaceable: layout1.start_offset() == 0,
            input2_inplaceable: layout2.start_offset() == 0,
        };

        if layout1.shape().elem_count() > 65535 * 64 {
            queue.add_const(candle_wgpu_kernels::Constants::UseZ, true);
        }

        queue.get_pipeline_const_inplace(
            Pipelines::Binary(
                dev.get_dtype(dtype)?,
                Functions::BinaryBufferFromBufferContiguousBoth,
            ),
            const_vec,
            inplaceable,
        )
    } else {
        let const_vec = vec![op as usize];
        queue.add_layout1(&layout1);

        if layout1 != layout2{
            queue.add_layout2(&layout2);

            if input1.layout().shape().elem_count() > 65535 * 64 {
                queue.add_const(candle_wgpu_kernels::Constants::UseZ, true);
            }

            queue.get_pipeline_const(
                Pipelines::Binary(dev.get_dtype(dtype)?, Functions::BinaryBufferFromBuffer),
                const_vec,
            )
        }
        else{
            if layout1.shape().elem_count() > 65535 * 64 {
                queue.add_const(candle_wgpu_kernels::Constants::UseZ, true);
            }

            queue.get_pipeline_const(
                Pipelines::Binary(dev.get_dtype(dtype)?, Functions::BinaryBufferFromBufferSameStride),
                const_vec,
            )
        }
    };

    let bind_group =
        dev.create_bind_group_input2(buffer_dest, input1.buffer(), input2.buffer(), dtype.into());

    queue.enqueue_64_big_extra(
        pipeline,
        bind_group,
        layout1.shape().elem_count() as u32,
        #[cfg(feature = "wgpu_debug")]
        Some(format!("OP: {:?}, layout1: {:?}, layout2: {:?}", op, layout1, layout2)),
    );
    Ok(())
}
