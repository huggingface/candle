pub mod err;
pub mod kernel;
pub mod kernels;
pub mod metal;
pub mod source;
pub mod utils;

pub use err::MetalKernelError;
pub use kernel::Kernels;
pub use kernels::{
    affine::*, call_binary_contiguous, call_binary_strided, call_gdn_causal_conv1d_output_f32,
    call_gdn_causal_conv1d_state_f32, call_gdn_chunked_scan_build_and_solve_f32,
    call_gdn_chunked_scan_solve_f32, call_gdn_decay_beta_gate_f32, call_gdn_decode_step_f32,
    call_gdn_l2_normalize_scale_f32, call_mlx_gemm, cast::*, convolution::*, fill::*, indexing::*,
    quantized::*, random::*, reduce::*, sdpa::*, sort::*, ternary::*, unary, unary::*, GemmDType,
    GgmlDType, GDN_SCAN_CHUNK,
};
use metal::{
    Buffer, CommandQueue, ComputeCommandEncoder, ComputePipeline, ConstantValues, Device, Function,
    Library, MTLResourceOptions, Value,
};
use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize};
use source::Source;
use utils::{get_block_dims, get_tile_size, linear_split, EncoderParam, EncoderProvider};
pub use utils::{BufferOffset, Output};

pub const RESOURCE_OPTIONS: MTLResourceOptions = objc2_metal::MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

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
