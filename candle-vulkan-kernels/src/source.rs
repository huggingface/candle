//! Compiled SPIR-V binaries, generated from `shaders/*.slang` by `build.rs` via slangc.

use crate::kernel::KernelName;

const AFFINE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/affine.spv"));
const GEMM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm.spv"));
const CONST_SET_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/const_set.spv"));
const UNARY_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/unary.spv"));
const BINARY_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary.spv"));
const CMP_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp.spv"));
const BINARY_BF16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary_bf16.spv"));
const CMP_BF16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_bf16.spv"));
const BINARY_F8E4M3_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary_f8e4m3.spv"));
const CMP_F8E4M3_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_f8e4m3.spv"));
const WHERE_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/where.spv"));
const COPY2D_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy2d.spv"));
const GATHER_IDX_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gather_idx.spv"));
const REDUCE_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reduce_ops.spv"));
const POOL2D_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pool2d.spv"));
const UPSAMPLE_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/upsample.spv"));
const CONV_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv.spv"));
const SCATTER_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scatter.spv"));

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
    Gemm,
    ConstSet,
    UnarySlang,
    BinarySlang,
    CmpSlang,
    BinaryBf16,
    CmpBf16,
    BinaryF8e4m3,
    CmpF8e4m3,
    WhereSlang,
    Copy2dSlang,
    GatherIdxSlang,
    ReduceSlang,
    Pool2dSlang,
    UpsampleSlang,
    ConvSlang,
    ScatterSlang,
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
            | KernelName::UnaryEluF32
            | KernelName::BinaryMaximumF32
            | KernelName::BinaryMinimumF32
            | KernelName::CmpEqF32
            | KernelName::CmpNeF32
            | KernelName::CmpLtF32
            | KernelName::CmpLeF32
            | KernelName::CmpGtF32
            | KernelName::CmpGeF32
            | KernelName::BinaryAddBf16
            | KernelName::BinarySubBf16
            | KernelName::BinaryMulBf16
            | KernelName::BinaryDivBf16
            | KernelName::BinaryMaximumBf16
            | KernelName::BinaryMinimumBf16
            | KernelName::CmpEqBf16
            | KernelName::CmpNeBf16
            | KernelName::CmpLtBf16
            | KernelName::CmpLeBf16
            | KernelName::CmpGtBf16
            | KernelName::CmpGeBf16
            | KernelName::BinaryAddF8e4m3
            | KernelName::BinarySubF8e4m3
            | KernelName::BinaryMulF8e4m3
            | KernelName::BinaryDivF8e4m3
            | KernelName::BinaryMaximumF8e4m3
            | KernelName::BinaryMinimumF8e4m3
            | KernelName::CmpEqF8e4m3
            | KernelName::CmpNeF8e4m3
            | KernelName::CmpLtF8e4m3
            | KernelName::CmpLeF8e4m3
            | KernelName::CmpGtF8e4m3
            | KernelName::CmpGeF8e4m3
            | KernelName::WhereF32
            | KernelName::Copy2dF32
            | KernelName::GatherIdxF32
            | KernelName::GatherRowsF32
            | KernelName::IndexSelectF32
            | KernelName::ReduceMinF32
            | KernelName::ReduceArgMinF32
            | KernelName::ReduceArgMaxF32
            | KernelName::ReduceSumF32
            | KernelName::ReduceMaxF32
            | KernelName::AffineF32
            | KernelName::BinaryAddF32
            | KernelName::BinarySubF32
            | KernelName::BinaryMulF32
            | KernelName::BinaryDivF32
            | KernelName::UnaryExpF32
            | KernelName::UnarySiluF32
            | KernelName::UnarySqrtF32
            | KernelName::UnarySinF32
            | KernelName::UnaryCosF32
            | KernelName::UnaryNegF32
            | KernelName::GemmF32
            | KernelName::AvgPool2dF32
            | KernelName::MaxPool2dF32
            | KernelName::UpsampleNearest1dF32
            | KernelName::UpsampleNearest2dF32
            | KernelName::UpsampleBilinear2dF32
            | KernelName::Conv1dF32
            | KernelName::Conv2dF32
            | KernelName::ConvTranspose1dF32
            | KernelName::ConvTranspose2dF32
            | KernelName::ScatterF32 => 0,
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
            Self::Gemm => GEMM_SPV,
            Self::ConstSet => CONST_SET_SPV,
            Self::UnarySlang => UNARY_SLANG_SPV,
            Self::BinarySlang => BINARY_SLANG_SPV,
            Self::CmpSlang => CMP_SLANG_SPV,
            Self::BinaryBf16 => BINARY_BF16_SPV,
            Self::CmpBf16 => CMP_BF16_SPV,
            Self::BinaryF8e4m3 => BINARY_F8E4M3_SPV,
            Self::CmpF8e4m3 => CMP_F8E4M3_SPV,
            Self::WhereSlang => WHERE_SLANG_SPV,
            Self::Copy2dSlang => COPY2D_SLANG_SPV,
            Self::GatherIdxSlang => GATHER_IDX_SLANG_SPV,
            Self::ReduceSlang => REDUCE_SLANG_SPV,
            Self::Pool2dSlang => POOL2D_SLANG_SPV,
            Self::UpsampleSlang => UPSAMPLE_SLANG_SPV,
            Self::ConvSlang => CONV_SLANG_SPV,
            Self::ScatterSlang => SCATTER_SLANG_SPV,
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
            Self::Gemm => "gemm",
            Self::ConstSet => "const_set",
            Self::UnarySlang => "unary",
            Self::BinarySlang => "binary",
            Self::CmpSlang => "cmp",
            Self::BinaryBf16 => "binary_bf16",
            Self::CmpBf16 => "cmp_bf16",
            Self::BinaryF8e4m3 => "binary_f8e4m3",
            Self::CmpF8e4m3 => "cmp_f8e4m3",
            Self::WhereSlang => "where",
            Self::Copy2dSlang => "copy2d",
            Self::GatherIdxSlang => "gather_idx",
            Self::ReduceSlang => "reduce",
            Self::Pool2dSlang => "pool2d",
            Self::UpsampleSlang => "upsample",
            Self::ConvSlang => "conv",
            Self::ScatterSlang => "scatter",
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
