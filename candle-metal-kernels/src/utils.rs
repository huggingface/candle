use crate::{Buffer, CommandBuffer, ComputeCommandEncoder, ComputePipeline};
use objc2_metal::MTLSize;
use std::ffi::c_void;

/// Most kernels apply similarly across the tensors
/// This creates a strategy that uses the maximum amount of threads per threadgroup (capped at the
/// actual total buffer length).
/// Then kernels can just do their op on their single point in the buffer.
pub(crate) fn linear_split(pipeline: &ComputePipeline, length: usize) -> (MTLSize, MTLSize) {
    let size = length;
    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), size);
    let count = size.div_ceil(width);
    let thread_group_count = MTLSize {
        width: count,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    (thread_group_count, thread_group_size)
}

// https://github.com/ml-explore/mlx/blob/bddf23f175726a57f0e443cd45518c0757daa166/mlx/backend/metal/utils.h#L96
pub fn get_block_dims(dim0: usize, dim1: usize, dim2: usize) -> MTLSize {
    let mut pows0 = 0;
    let mut pows1 = 0;
    let mut pows2 = 0;
    let mut sum = 0;
    loop {
        let presum = sum;
        // Check all the pows
        if dim0 >= (1 << (pows0 + 1)) {
            pows0 += 1;
            sum += 1;
        }
        if sum == 10 {
            break;
        }
        if dim1 >= (1 << (pows1 + 1)) {
            pows1 += 1;
            sum += 1;
        }
        if sum == 10 {
            break;
        }
        if dim2 >= (1 << (pows2 + 1)) {
            pows2 += 1;
            sum += 1;
        }
        if sum == presum || sum == 10 {
            break;
        }
    }
    MTLSize {
        width: 1 << pows0,
        height: 1 << pows1,
        depth: 1 << pows2,
    }
}

pub fn set_param<P: EncoderParam>(encoder: &ComputeCommandEncoder, position: usize, data: P) {
    <P as EncoderParam>::set_param(encoder, position, data)
}

/// Helper functions to create the various objects on the compute command encoder
/// on a single line.
/// Prevents getting wrong some arguments number and mixing length and size in bytes.
pub trait EncoderParam {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self);
}
macro_rules! primitive {
    ($type:ty) => {
        impl EncoderParam for $type {
            fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
                encoder.set_bytes(
                    position,
                    core::mem::size_of::<$type>(),
                    &data as *const $type as *const c_void,
                );
            }
        }
    };
}
primitive!(bool);
primitive!(usize);
primitive!(i32);
primitive!(i64);
primitive!(u8);
primitive!(u32);
primitive!(u64);
primitive!(f32);
primitive!(f64);
primitive!(half::bf16);
primitive!(half::f16);

pub struct BufferOffset<'a> {
    pub buffer: &'a Buffer,
    pub offset_in_bytes: usize,
}

impl<'a> BufferOffset<'a> {
    pub fn zero_offset(buffer: &'a Buffer) -> Self {
        Self {
            buffer,
            offset_in_bytes: 0,
        }
    }
}

impl<T> EncoderParam for &[T] {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes(position, core::mem::size_of_val(data), data.as_ptr().cast());
    }
}

impl EncoderParam for &Buffer {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}

impl EncoderParam for (&Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1);
    }
}

impl EncoderParam for &BufferOffset<'_> {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data.buffer), data.offset_in_bytes);
    }
}

impl EncoderParam for &mut Buffer {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}

impl EncoderParam for (&mut Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1);
    }
}

#[macro_export]
macro_rules! set_params {
    ($encoder:ident, ($($param:expr),+)) => (
        let mut _index = 0;
        $(
            $crate::utils::set_param($encoder, _index, $param);
            _index += 1;
        )*
    );
}

pub trait EncoderProvider {
    type Encoder<'a>: AsRef<ComputeCommandEncoder>
    where
        Self: 'a;

    const END_ENCODING_ON_DROP: bool;

    fn encoder(&self) -> Self::Encoder<'_>;
}

pub struct WrappedEncoder<'a> {
    inner: &'a ComputeCommandEncoder,
    end_encoding_on_drop: bool,
}

impl Drop for WrappedEncoder<'_> {
    fn drop(&mut self) {
        if self.end_encoding_on_drop {
            self.inner.end_encoding()
        }
    }
}

impl AsRef<ComputeCommandEncoder> for WrappedEncoder<'_> {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self.inner
    }
}

impl EncoderProvider for &CommandBuffer {
    const END_ENCODING_ON_DROP: bool = true;
    type Encoder<'a>
        = ComputeCommandEncoder
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        self.compute_command_encoder()
    }
}

impl EncoderProvider for &ComputeCommandEncoder {
    const END_ENCODING_ON_DROP: bool = false;
    type Encoder<'a>
        = WrappedEncoder<'a>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        WrappedEncoder {
            inner: self,
            end_encoding_on_drop: false,
        }
    }
}
