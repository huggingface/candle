//! Compiled SPIR-V binaries, generated from `shaders/*.comp` by `build.rs` via naga.

use crate::kernel::KernelName;

const AFFINE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/affine.spv"));
const ELEMENTWISE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/elementwise.spv"));
const GEMM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm.spv"));
const REDUCE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reduce.spv"));
const REDUCE_MAX_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reduce_max.spv"));
const GATHER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gather.spv"));
const COPY_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy.spv"));

/// The set of compiled compute shaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    Elementwise,
    Gemm,
    Reduce,
    ReduceMax,
    Gather,
    Copy,
}

impl Source {
    /// Size of the push constant block for this shader entry point, in
    /// bytes. The pipeline layout must declare exactly the bytes the shader
    /// accesses, otherwise dispatch fails validation.
    pub fn push_constant_size(self, name: KernelName) -> u32 {
        match name {
            KernelName::AffineF32 => 12,
            KernelName::ReduceSumF32 => 8,
            KernelName::ReduceMaxF32 => 8,
            KernelName::GatherF32 => 8,
            KernelName::CopyF32 => 64,
            KernelName::ElemAddF32
            | KernelName::ElemSubF32
            | KernelName::ElemMulF32
            | KernelName::ElemDivF32 => 44,
            KernelName::ElemSigmoidF32
            | KernelName::ElemSiluF32
            | KernelName::ElemExpF32
            | KernelName::ElemSqrtF32
            | KernelName::ElemSinF32
            | KernelName::ElemCosF32
            | KernelName::ElemNegF32 => 44,
            KernelName::GemmF32 => 20,
        }
    }

    /// Raw SPIR-V words (little-endian u32) for this shader.
    pub fn spv_words(self) -> Vec<u32> {
        let bytes: &[u8] = match self {
            Self::Affine => AFFINE_SPV,
            Self::Elementwise => ELEMENTWISE_SPV,
            Self::Gemm => GEMM_SPV,
            Self::Reduce => REDUCE_SPV,
            Self::ReduceMax => REDUCE_MAX_SPV,
            Self::Gather => GATHER_SPV,
            Self::Copy => COPY_SPV,
        };
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}
