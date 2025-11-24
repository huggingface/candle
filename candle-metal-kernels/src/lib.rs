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
    fill::*, indexing::*, quantized::*, random::*, reduce::*, sdpa::*, sort::*, ternary::*, unary,
    unary::*, GemmDType, GgmlDType,
};
use metal::{
    BlitCommandEncoder, Buffer, CommandQueue, ComputeCommandEncoder, ComputePipeline,
    ConstantValues, Device, Function, Library, MTLResourceOptions, Value,
};
use objc2_metal::{MTLCompileOptions, MTLMathMode, MTLSize};
use source::Source;
pub use utils::BufferOffset;
use utils::{get_block_dims, linear_split, EncoderParam, EncoderProvider};

/// Default resource options for performance-critical GPU buffers.
/// Currently uses Private storage; can be tuned or made configurable in the future.
pub const RESOURCE_OPTIONS: MTLResourceOptions =
    objc2_metal::MTLResourceOptions(MTLResourceOptions::StorageModePrivate.bits());

/// Resource options for CPU-visible/readback buffers (tests, diagnostics, etc.).
/// Using Shared storage ensures `contents()` can be safely used.
pub const RESOURCE_OPTIONS_SHARED: MTLResourceOptions =
    objc2_metal::MTLResourceOptions(MTLResourceOptions::StorageModeShared.bits());

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

#[cfg(test)]
mod tests;
