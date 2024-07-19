use std::sync::Arc;

use rand::RngCore;

use crate::{wgpu::{cache::BufferReference, device::Pipelines}, WgpuDevice};

use super::{create_bind_group_input0, create_bind_group_input1, enqueue_big, get_meta};


#[derive(Copy, Clone, Debug)]
//#[allow(dead_code)]
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
    buffer: Arc<BufferReference>,
    op: UnaryOperation,
    scalar1: f32,
    scalar2: f32,
    dtype: crate::DType,
    layout: crate::Layout,
) -> crate::Result<()> {

    if layout.is_contiguous() {
        let mut meta = get_meta(&dev);
        meta.add(op as u32);
        meta.add(scalar1);
        meta.add(scalar2);
        meta.add(dev.rand_state.lock().unwrap().next_u32());
        meta.add(layout.start_offset());       //offset, do not change, will be checked at flush_gpu_command
        meta.add(layout.shape().elem_count()); //length 

        let pipeline = dev.get_pipeline(super::Shader::Unary(dtype), Pipelines::UnaryInplaceContiguous)?;

        let bind_group = create_bind_group_input0( buffer);
        enqueue_big(
            meta,
            pipeline,
            bind_group,
            layout.shape().elem_count() as u32,
            #[cfg(feature = "wgpu_debug")] 
            crate::wgpu::device::QueueDebugInfo::new(&format!("unary_inplace_contiguoes op:{:?}, dtype:{:?}, pipeline:{:?}",op, dtype, Pipelines::UnaryInplaceContiguous)),
        );
    }
    else{
        panic!("can only query unary inplace for contigueos memory!");
    }
    return Ok(());
}

pub fn queue_unary_from_buffer_op(
    dev: &WgpuDevice,
    buffer_dest: Arc<BufferReference>,
    buffer_input: Arc<BufferReference>,
    op: UnaryOperation,
    scalar1: f32,
    scalar2: f32,
    dtype: crate::DType,
    input_layout: &crate::Layout,
) -> crate::Result<()> {
    if input_layout.is_contiguous() {
        let mut meta = get_meta(&dev);
        meta.add(op as u32);
        meta.add(scalar1);
        meta.add(scalar2);
        meta.add(dev.rand_state.lock().unwrap().next_u32());
        meta.add(input_layout.start_offset());       //
        meta.add(input_layout.shape().elem_count()); //length 

        let pipeline_type = 
        if input_layout.start_offset() == 0{
            Pipelines::UnaryFromBufferContiguousNoStartOffset
        }
        else{
            Pipelines::UnaryFromBufferContiguous
        };
        let pipeline = dev.get_pipeline(super::Shader::Unary(dtype), pipeline_type.clone())?;

        let bind_group = create_bind_group_input1( buffer_dest, buffer_input);
        enqueue_big(
            meta,
            pipeline,
            bind_group,
            input_layout.shape().elem_count() as u32,
            #[cfg(feature = "wgpu_debug")] 
            crate::wgpu::device::QueueDebugInfo::new(&format!("unary op:{:?}, dtype:{:?}, pipeline:{:?}",op, dtype, pipeline_type)),
        );
    } else {
        let mut meta = get_meta(&dev);
        meta.add(op as u32);
        meta.add(scalar1);
        meta.add(scalar2);
        meta.add(dev.rand_state.lock().unwrap().next_u32());
        meta.add_layout(&input_layout);

        let pipeline = dev.get_pipeline(super::Shader::Unary(dtype), Pipelines::UnaryFromBuffer)?;
        let bind_group = create_bind_group_input1( buffer_dest, buffer_input);
        enqueue_big(
            meta,
            pipeline,
            bind_group,
            input_layout.shape().elem_count() as u32,
            #[cfg(feature = "wgpu_debug")] 
            crate::wgpu::device::QueueDebugInfo::new(&format!("unary op:{:?}, dtype:{:?}, pipeline:{:?}",op, dtype, Pipelines::UnaryFromBuffer)),
        );
    }
    return Ok(());
}
