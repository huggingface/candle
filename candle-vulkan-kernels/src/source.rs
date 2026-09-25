//! Compiled SPIR-V binaries, generated from `shaders/*.comp` by `build.rs` via naga.

use crate::kernel::KernelName;

const AFFINE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/affine.spv"));
const ELEMENTWISE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/elementwise.spv"));
const GEMM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm.spv"));
const REDUCE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reduce.spv"));
const REDUCE_MAX_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reduce_max.spv"));
const GATHER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gather.spv"));
const COPY_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy.spv"));
const RMS_NORM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm.spv"));
const SOFTMAX_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/softmax.spv"));
const ROPE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv"));
const GEMV_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv.spv"));
const GEMV_T_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_t.spv"));
const Q4K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4k.spv"));
const Q5Q8_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q5q8.spv"));
const CONST_SET_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/const_set.spv"));
const UNARY_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/unary.spv"));
const BINARY_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary.spv"));
const CMP_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp.spv"));
const WHERE_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/where.spv"));

const TEST_FILL_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_f16.spv"));
const TEST_FILL_U8_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_u8.spv"));
const TEST_FILL_I16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_i16.spv"));
const TEST_FILL_BF16_NATIVE_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_native_bf16.spv"));
const TEST_FILL_BF16_EMULATED_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_emulated_bf16.spv"));
const TEST_FILL_U32_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_u32.spv"));
const TEST_FILL_I32_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_i32.spv"));
const TEST_FILL_F32_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_f32.spv"));
const TEST_FILL_I64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_i64.spv"));
const TEST_FILL_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_f64.spv"));
const TEST_FILL_F4_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_f4.spv"));
const TEST_FILL_F6E2M3_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_f6e2m3.spv"));
const TEST_FILL_F6E3M2_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_f6e3m2.spv"));
const TEST_FILL_F8E4M3_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_f8e4m3.spv"));
const TEST_FILL_F8E8M0_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/test/fill_f8e8m0.spv"));

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
    RmsNorm,
    Softmax,
    Rope,
    Gemv,
    GemvT,
    Q4k,
    Q5q8,
    ConstSet,
    UnarySlang,
    BinarySlang,
    CmpSlang,
    WhereSlang,
    TestFillF16,
    TestFillU8,
    TestFillI16,
    TestFillBf16Native,
    TestFillBf16Emulated,
    TestFillU32,
    TestFillI32,
    TestFillF32,
    TestFillI64,
    TestFillF64,
    TestFillF4,
    TestFillF6e2m3,
    TestFillF6e3m2,
    TestFillF8e4m3,
    TestFillF8e8m0,
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
            KernelName::RmsNormF32 => 12,
            KernelName::SoftmaxLastDimF32 => 8,
            KernelName::RopeF32 => 20,
            KernelName::GemvF32 => 8,
            KernelName::GemvTF32 => 12,
            KernelName::Q4kQmatvecF32 => 8,
            KernelName::Q4kDequantF32 => 8,
            KernelName::Q6kDequantF32 => 8,
            KernelName::Q80DequantF32 => 8,
            KernelName::Q50DequantF32 => 8,
            KernelName::Q5KDequantF32 => 8,
            KernelName::ConstSetF32 => 0,
            KernelName::UnaryLogF32
            | KernelName::UnaryAbsF32
            | KernelName::UnaryRecipF32
            | KernelName::UnarySqrF32
            | KernelName::UnaryGeluF32
            | KernelName::UnaryGeluErfF32
            | KernelName::UnaryErfF32
            | KernelName::UnaryReluF32
            | KernelName::UnaryTanhF32
            | KernelName::UnaryFloorF32
            | KernelName::UnaryCeilF32
            | KernelName::UnaryRoundF32
            | KernelName::UnarySignF32
            | KernelName::UnaryPowF32
            | KernelName::BinaryMaximumF32
            | KernelName::BinaryMinimumF32
            | KernelName::CmpEqF32
            | KernelName::CmpNeF32
            | KernelName::CmpLtF32
            | KernelName::CmpLeF32
            | KernelName::CmpGtF32
            | KernelName::CmpGeF32
            | KernelName::WhereF32 => 0,
            KernelName::TestFillF16 => 0,
            KernelName::TestFillU8
            | KernelName::TestFillI16
            | KernelName::TestFillBf16Native
            | KernelName::TestFillBf16Emulated
            | KernelName::TestFillU32
            | KernelName::TestFillI32
            | KernelName::TestFillF32
            | KernelName::TestFillI64
            | KernelName::TestFillF4
            | KernelName::TestFillF6e2m3
            | KernelName::TestFillF6e3m2
            | KernelName::TestFillF8e4m3
            | KernelName::TestFillF8e8m0
            | KernelName::TestFillF64 => 0,
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
            Self::RmsNorm => RMS_NORM_SPV,
            Self::Softmax => SOFTMAX_SPV,
            Self::Rope => ROPE_SPV,
            Self::Gemv => GEMV_SPV,
            Self::GemvT => GEMV_T_SPV,
            Self::Q4k => Q4K_SPV,
            Self::Q5q8 => Q5Q8_SPV,
            Self::ConstSet => CONST_SET_SPV,
            Self::UnarySlang => UNARY_SLANG_SPV,
            Self::BinarySlang => BINARY_SLANG_SPV,
            Self::CmpSlang => CMP_SLANG_SPV,
            Self::WhereSlang => WHERE_SLANG_SPV,
            Self::TestFillF16 => TEST_FILL_F16_SPV,
            Self::TestFillU8 => TEST_FILL_U8_SPV,
            Self::TestFillI16 => TEST_FILL_I16_SPV,
            Self::TestFillBf16Native => TEST_FILL_BF16_NATIVE_SPV,
            Self::TestFillBf16Emulated => TEST_FILL_BF16_EMULATED_SPV,
            Self::TestFillU32 => TEST_FILL_U32_SPV,
            Self::TestFillI32 => TEST_FILL_I32_SPV,
            Self::TestFillF32 => TEST_FILL_F32_SPV,
            Self::TestFillI64 => TEST_FILL_I64_SPV,
            Self::TestFillF64 => TEST_FILL_F64_SPV,
            Self::TestFillF4 => TEST_FILL_F4_SPV,
            Self::TestFillF6e2m3 => TEST_FILL_F6E2M3_SPV,
            Self::TestFillF6e3m2 => TEST_FILL_F6E3M2_SPV,
            Self::TestFillF8e4m3 => TEST_FILL_F8E4M3_SPV,
            Self::TestFillF8e8m0 => TEST_FILL_F8E8M0_SPV,
        };
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

impl AsRef<str> for Source {
    fn as_ref(&self) -> &str {
        match self {
            Self::Affine => "affine",
            Self::Elementwise => "elementwise",
            Self::Gemm => "gemm",
            Self::Gemv => "gemv",
            Self::GemvT => "gemv_t",
            Self::Q4k => "q4k",
            Self::Q5q8 => "q5q8",
            Self::Reduce => "reduce",
            Self::ReduceMax => "reduce_max",
            Self::Gather => "gather",
            Self::Copy => "copy",
            Self::RmsNorm => "rms_norm",
            Self::Softmax => "softmax",
            Self::Rope => "rope",
            Self::ConstSet => "const_set",
            Self::UnarySlang => "unary",
            Self::BinarySlang => "binary",
            Self::CmpSlang => "cmp",
            Self::WhereSlang => "where",
            Self::TestFillF16 => "test_fill_f16",
            Self::TestFillU8 => "test_fill_u8",
            Self::TestFillI16 => "test_fill_i16",
            Self::TestFillBf16Native => "test_fill_bf16_native",
            Self::TestFillBf16Emulated => "test_fill_bf16_emulated",
            Self::TestFillU32 => "test_fill_u32",
            Self::TestFillI32 => "test_fill_i32",
            Self::TestFillF32 => "test_fill_f32",
            Self::TestFillI64 => "test_fill_i64",
            Self::TestFillF64 => "test_fill_f64",
            Self::TestFillF4 => "test_fill_f4",
            Self::TestFillF6e2m3 => "test_fill_f6e2m3",
            Self::TestFillF6e3m2 => "test_fill_f6e3m2",
            Self::TestFillF8e4m3 => "test_fill_f8e4m3",
            Self::TestFillF8e8m0 => "test_fill_f8e8m0",
        }
    }
}
