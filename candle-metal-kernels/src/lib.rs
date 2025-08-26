pub mod err;
pub mod kernel;
pub mod kernels;
pub mod metal;
pub mod source;
pub mod utils;

pub use err::MetalKernelError;
pub use kernel::Kernels;
pub use kernels::{
    affine::*, call_binary_contiguous, call_binary_strided, call_mlx_gemm, cast::*, convolution::*,
    indexing::*, quantized::*, random::*, reduce::*, sort::*, ternary::*, unary, unary::*,
    GemmDType, GgmlDType,
};
use metal::{
    BlitCommandEncoder, Buffer, CommandQueue, ComputeCommandEncoder, ComputePipeline,
    ConstantValues, Device, Function, Library, MTLResourceOptions, Value,
};
use objc2_metal::{MTLCompileOptions, MTLMathMode, MTLResourceUsage, MTLSize};
use source::Source;
pub use utils::BufferOffset;
use utils::{get_block_dims, linear_split, EncoderParam, EncoderProvider};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    BF16,
    F16,
    F32,
    I64,
    U32,
    U8,
}

impl DType {
    fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

pub fn call_const_fill(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    length: usize,
    output: &Buffer,
    v: impl EncoderParam,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Fill, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(encoder, (output, v, length));
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[cfg(test)]
mod tests;
