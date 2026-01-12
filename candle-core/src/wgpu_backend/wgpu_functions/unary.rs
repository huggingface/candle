use super::*;
use candle_wgpu_kernels::unary::Functions;
use rand::RngCore;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum UnaryOperation {
    SetZero = 0,
    SetOne = 1,
    IncOne = 2,
    DecOne = 3,
    Identity = 4,
    Square = 5,
    Affine = 6,
    Abs = 7,
    Acos = 8,
    Acosh = 9,
    Asin = 10,
    Asinh = 11,
    Atan = 12,
    Atanh = 13,
    Ceil = 14,
    Cos = 15,
    Cosh = 16,
    Deg = 17,
    Exp = 21,
    Floor = 22,
    Fract = 23,
    InverseSqrt = 24,
    Log = 25,
    Log2 = 26,
    Rad = 27,
    Sign = 28,
    Sin = 29,
    Sinh = 31,
    Sqrt = 32,
    Tan = 33,
    Tanh = 34,
    Trunc = 35,
    BinaryStep = 36,
    Sigmoid = 37,
    Relu = 38,
    Softplus = 39,
    LeakyRelu = 40,
    SiLu = 41,
    Gassian = 42,

    Neg = 45,
    Inverse = 46,
    RandNormal = 47,
    RandUniform = 48,
    Gelu = 49,
    Round = 50,
    Elu = 52,
    Erf = 53,
    GeluErf = 54,

    SetScalar = 100,
    AddScalar = 101,
    MultScalar = 102,
    MinusScalar = 103,
    DivScalar = 104,
    MaxScalar = 105,
    MinScalar = 106,
    PowScalar = 107,
}

pub fn queue_unary_inplace_op(
    dev: &WgpuDevice,
    buffer: BufferReferenceId,
    op: UnaryOperation,
    scalar1: f32,
    scalar2: f32,
    dtype: crate::DType,
    layout: &crate::Layout,
) -> crate::Result<()> {
    if layout.is_contiguous() {
        let const_vec = vec![op as u32, (layout.start_offset() == 0) as u32];

        let mut queue = dev.get_queue();
        queue.add(scalar1);
        queue.add(scalar2);
        queue.add(layout.shape().elem_count()); //length

        let mut is_contiguous4 = false;
        let pipeline = match op {
            UnaryOperation::SetZero | UnaryOperation::SetOne => {
                if layout.shape().elem_count().is_multiple_of(4) && dtype.size_in_bytes() == 4 {
                    is_contiguous4 = true;
                    Pipelines::Unary(dev.get_dtype(dtype)?, Functions::ConstInplaceContiguous4)
                } else {
                    Pipelines::Unary(dev.get_dtype(dtype)?, Functions::ConstInplaceContiguous)
                }
            }
            UnaryOperation::RandNormal | UnaryOperation::RandUniform => {
                Pipelines::Unary(dev.get_dtype(dtype)?, Functions::RandInplaceContiguous)
            }
            _ => Pipelines::Unary(dev.get_dtype(dtype)?, Functions::UnaryInplaceContiguous),
        };

        if layout.start_offset() != 0
            || op == UnaryOperation::RandNormal
            || op == UnaryOperation::RandUniform
        {
            if is_contiguous4 {
                queue.add(layout.start_offset() / 4);
            } else {
                queue.add(layout.start_offset());
            }
        }
        if op == UnaryOperation::RandNormal || op == UnaryOperation::RandUniform {
            queue.add(dev.inner_device().rand_state.lock().unwrap().next_u32());
        }

        let length = if is_contiguous4 {
            (layout.shape().elem_count() / 4) as u32
        } else {
            layout.shape().elem_count() as u32
        };

        if length > 65535 * 64 {
            queue.add_const(candle_wgpu_kernels::Constants::UseZ, true);
        }

        let pipeline = queue.get_pipeline_const(pipeline, const_vec);

        let bind_group = dev.create_bind_group_input0(
            buffer,
            if is_contiguous4 {
                BindgroupAlignment::Aligned16
            } else {
                dtype.into()
            },
        );

        queue.enqueue_64_big_extra(
            pipeline,
            bind_group,
            length,
            #[cfg(feature = "wgpu_debug")]
            Some(format!("OP: {:?}, layout: {:?}", op, layout)),
        );
    } else {
        let const_vec = vec![op as u32];

        let mut queue = dev.get_queue();
        queue.add(scalar1);
        queue.add(scalar2);
        queue.add_layout1(layout);

        let pipeline = match op {
            UnaryOperation::SetZero | UnaryOperation::SetOne => {
                Pipelines::Unary(dev.get_dtype(dtype)?, Functions::ConstInplaceNonContiguous)
            }
            _ => Pipelines::Unary(dev.get_dtype(dtype)?, Functions::UnaryInplaceNonContiguous),
        };

        let length = layout.shape().elem_count() as u32;

        let pipeline = queue.get_pipeline_const(pipeline, const_vec);

        let bind_group = dev.create_bind_group_input0(buffer, dtype.into());

        queue.enqueue_64_big_extra(
            pipeline,
            bind_group,
            length,
            #[cfg(feature = "wgpu_debug")]
            Some(format!("OP: {:?}, layout: {:?}", op, layout)),
        );
    }
    Ok(())
}

pub fn queue_unary_from_buffer_op(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    input: WgpuTensor,
    op: UnaryOperation,
    scalar1: f32,
    scalar2: f32,
    dtype: crate::DType,
) -> crate::Result<()> {
    let mut queue = dev.get_queue();
    let pipeline = if input.layout().is_contiguous() {
        let const_vec = vec![op as u32, (input.layout().start_offset() == 0) as u32];

        queue.add(scalar1);
        queue.add(scalar2);
        queue.add(input.layout().shape().elem_count()); //length

        if input.layout().start_offset() != 0
            || op == UnaryOperation::RandNormal
            || op == UnaryOperation::RandUniform
        {
            queue.add(input.layout().start_offset());
        }
        if op == UnaryOperation::RandNormal || op == UnaryOperation::RandUniform {
            queue.add(dev.inner_device().rand_state.lock().unwrap().next_u32());
        }

        let inplaceable = OpIsInplaceable {
            input1_inplaceable: input.layout().start_offset() == 0,
            input2_inplaceable: false,
        };

        if input.layout().shape().elem_count() > 65535 * 64 {
            queue.add_const(candle_wgpu_kernels::Constants::UseZ, true);
        }

        queue.get_pipeline_const_inplace(
            Pipelines::Unary(dev.get_dtype(dtype)?, Functions::UnaryFromBufferContiguous),
            const_vec,
            inplaceable,
        )
    } else {
        let const_vec = vec![op as u32];

        queue.add(scalar1);
        queue.add(scalar2);
        queue.add_layout1(input.layout());

        if input.layout().shape().elem_count() > 65535 * 64 {
            queue.add_const(candle_wgpu_kernels::Constants::UseZ, true);
        }

        queue.get_pipeline_const(
            Pipelines::Unary(dev.get_dtype(dtype)?, Functions::UnaryFromBuffer),
            const_vec,
        )
    };

    let bind_group = dev.create_bind_group_input1(buffer_dest, input.buffer(), dtype.into());
    queue.enqueue_64_big_extra(
        pipeline,
        bind_group,
        input.layout().shape().elem_count() as u32,
        #[cfg(feature = "wgpu_debug")]
        Some(format!("OP: {:?}, layout: {:?}", op, input.layout())),
    );

    Ok(())
}
