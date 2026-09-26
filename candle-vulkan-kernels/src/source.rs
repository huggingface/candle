//! Compiled SPIR-V binaries, generated from `shaders/*.slang` by `build.rs` via slangc.

use crate::kernel::KernelName;

const AFFINE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/affine.spv"));
const GEMM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm.spv"));
const GEMM_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_f16.spv"));
const GEMM_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_f64.spv"));
const CONST_SET_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/const_set.spv"));
const UNARY_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/unary.spv"));
const BINARY_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary.spv"));
const CMP_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp.spv"));
const BINARY_BF16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary_bf16.spv"));
const CMP_BF16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_bf16.spv"));
const BINARY_F8E4M3_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary_f8e4m3.spv"));
const CMP_F8E4M3_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_f8e4m3.spv"));
const UNARY_BF16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/unary_bf16.spv"));
const UNARY_F8E4M3_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/unary_f8e4m3.spv"));
const WHERE_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/where.spv"));
const WHERE_BF16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/where_bf16.spv"));
const WHERE_F8E4M3_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/where_f8e4m3.spv"));
const AFFINE_BF16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/affine_bf16.spv"));
const AFFINE_F8E4M3_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/affine_f8e4m3.spv"));
const BINARY_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary_f16.spv"));
const CMP_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_f16.spv"));
const UNARY_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/unary_f16.spv"));
const AFFINE_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/affine_f16.spv"));
const WHERE_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/where_f16.spv"));
const BINARY_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary_f64.spv"));
const CMP_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_f64.spv"));
const UNARY_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/unary_f64.spv"));
const AFFINE_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/affine_f64.spv"));
const WHERE_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/where_f64.spv"));
const BINARY_U8_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary_u8.spv"));
const CMP_U8_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_u8.spv"));
const BINARY_U32_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary_u32.spv"));
const CMP_U32_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_u32.spv"));
const BINARY_I16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary_i16.spv"));
const CMP_I16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_i16.spv"));
const BINARY_I32_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary_i32.spv"));
const CMP_I32_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_i32.spv"));
const BINARY_I64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/binary_i64.spv"));
const CMP_I64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cmp_i64.spv"));
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
    GemmF16,
    GemmF64,
    ConstSet,
    UnarySlang,
    BinarySlang,
    CmpSlang,
    BinaryBf16,
    CmpBf16,
    BinaryF8e4m3,
    CmpF8e4m3,
    UnaryBf16,
    UnaryF8e4m3,
    WhereSlang,
    WhereBf16,
    WhereF8e4m3,
    AffineBf16,
    AffineF8e4m3,
    BinaryF16,
    CmpF16,
    UnaryF16,
    AffineF16,
    WhereF16,
    BinaryF64,
    CmpF64,
    UnaryF64,
    AffineF64,
    WhereF64,
    BinaryU8,
    CmpU8,
    BinaryU32,
    CmpU32,
    BinaryI16,
    CmpI16,
    BinaryI32,
    CmpI32,
    BinaryI64,
    CmpI64,
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
            | KernelName::UnaryLogBf16
            | KernelName::UnaryAbsBf16
            | KernelName::UnaryRecipBf16
            | KernelName::UnarySqrBf16
            | KernelName::UnaryGeluBf16
            | KernelName::UnaryGeluErfBf16
            | KernelName::UnaryErfBf16
            | KernelName::UnaryReluBf16
            | KernelName::UnaryTanhBf16
            | KernelName::UnaryFloorBf16
            | KernelName::UnaryCeilBf16
            | KernelName::UnaryRoundBf16
            | KernelName::UnarySignBf16
            | KernelName::UnaryExpBf16
            | KernelName::UnarySiluBf16
            | KernelName::UnarySqrtBf16
            | KernelName::UnarySinBf16
            | KernelName::UnaryCosBf16
            | KernelName::UnaryNegBf16
            | KernelName::UnaryLogF8e4m3
            | KernelName::UnaryAbsF8e4m3
            | KernelName::UnaryRecipF8e4m3
            | KernelName::UnarySqrF8e4m3
            | KernelName::UnaryGeluF8e4m3
            | KernelName::UnaryGeluErfF8e4m3
            | KernelName::UnaryErfF8e4m3
            | KernelName::UnaryReluF8e4m3
            | KernelName::UnaryTanhF8e4m3
            | KernelName::UnaryFloorF8e4m3
            | KernelName::UnaryCeilF8e4m3
            | KernelName::UnaryRoundF8e4m3
            | KernelName::UnarySignF8e4m3
            | KernelName::UnaryExpF8e4m3
            | KernelName::UnarySiluF8e4m3
            | KernelName::UnarySqrtF8e4m3
            | KernelName::UnarySinF8e4m3
            | KernelName::UnaryCosF8e4m3
            | KernelName::UnaryNegF8e4m3
            | KernelName::WhereF32
            | KernelName::WhereBf16
            | KernelName::WhereF8e4m3
            | KernelName::AffineBf16
            | KernelName::AffineF8e4m3
            | KernelName::UnaryPowBf16
            | KernelName::UnaryEluBf16
            | KernelName::UnaryPowF8e4m3
            | KernelName::UnaryEluF8e4m3
            | KernelName::BinaryAddF16
            | KernelName::BinarySubF16
            | KernelName::BinaryMulF16
            | KernelName::BinaryDivF16
            | KernelName::BinaryMaximumF16
            | KernelName::BinaryMinimumF16
            | KernelName::CmpEqF16
            | KernelName::CmpNeF16
            | KernelName::CmpLtF16
            | KernelName::CmpLeF16
            | KernelName::CmpGtF16
            | KernelName::CmpGeF16
            | KernelName::UnaryLogF16
            | KernelName::UnaryAbsF16
            | KernelName::UnaryRecipF16
            | KernelName::UnarySqrF16
            | KernelName::UnaryGeluF16
            | KernelName::UnaryGeluErfF16
            | KernelName::UnaryErfF16
            | KernelName::UnaryReluF16
            | KernelName::UnaryTanhF16
            | KernelName::UnaryFloorF16
            | KernelName::UnaryCeilF16
            | KernelName::UnaryRoundF16
            | KernelName::UnarySignF16
            | KernelName::UnaryPowF16
            | KernelName::UnaryEluF16
            | KernelName::UnaryExpF16
            | KernelName::UnarySiluF16
            | KernelName::UnarySqrtF16
            | KernelName::UnarySinF16
            | KernelName::UnaryCosF16
            | KernelName::UnaryNegF16
            | KernelName::AffineF16
            | KernelName::WhereF16
            | KernelName::BinaryAddF64
            | KernelName::BinarySubF64
            | KernelName::BinaryMulF64
            | KernelName::BinaryDivF64
            | KernelName::BinaryMaximumF64
            | KernelName::BinaryMinimumF64
            | KernelName::CmpEqF64
            | KernelName::CmpNeF64
            | KernelName::CmpLtF64
            | KernelName::CmpLeF64
            | KernelName::CmpGtF64
            | KernelName::CmpGeF64
            | KernelName::UnaryLogF64
            | KernelName::UnaryAbsF64
            | KernelName::UnaryRecipF64
            | KernelName::UnarySqrF64
            | KernelName::UnaryGeluF64
            | KernelName::UnaryGeluErfF64
            | KernelName::UnaryErfF64
            | KernelName::UnaryReluF64
            | KernelName::UnaryTanhF64
            | KernelName::UnaryFloorF64
            | KernelName::UnaryCeilF64
            | KernelName::UnaryRoundF64
            | KernelName::UnarySignF64
            | KernelName::UnaryPowF64
            | KernelName::UnaryEluF64
            | KernelName::UnaryExpF64
            | KernelName::UnarySiluF64
            | KernelName::UnarySqrtF64
            | KernelName::UnarySinF64
            | KernelName::UnaryCosF64
            | KernelName::UnaryNegF64
            | KernelName::AffineF64
            | KernelName::WhereF64
            | KernelName::BinaryAddU8
            | KernelName::BinarySubU8
            | KernelName::BinaryMulU8
            | KernelName::BinaryDivU8
            | KernelName::BinaryMaximumU8
            | KernelName::BinaryMinimumU8
            | KernelName::CmpEqU8
            | KernelName::CmpNeU8
            | KernelName::CmpLtU8
            | KernelName::CmpLeU8
            | KernelName::CmpGtU8
            | KernelName::CmpGeU8
            | KernelName::BinaryAddU32
            | KernelName::BinarySubU32
            | KernelName::BinaryMulU32
            | KernelName::BinaryDivU32
            | KernelName::BinaryMaximumU32
            | KernelName::BinaryMinimumU32
            | KernelName::CmpEqU32
            | KernelName::CmpNeU32
            | KernelName::CmpLtU32
            | KernelName::CmpLeU32
            | KernelName::CmpGtU32
            | KernelName::CmpGeU32
            | KernelName::BinaryAddI16
            | KernelName::BinarySubI16
            | KernelName::BinaryMulI16
            | KernelName::BinaryDivI16
            | KernelName::BinaryMaximumI16
            | KernelName::BinaryMinimumI16
            | KernelName::CmpEqI16
            | KernelName::CmpNeI16
            | KernelName::CmpLtI16
            | KernelName::CmpLeI16
            | KernelName::CmpGtI16
            | KernelName::CmpGeI16
            | KernelName::BinaryAddI32
            | KernelName::BinarySubI32
            | KernelName::BinaryMulI32
            | KernelName::BinaryDivI32
            | KernelName::BinaryMaximumI32
            | KernelName::BinaryMinimumI32
            | KernelName::CmpEqI32
            | KernelName::CmpNeI32
            | KernelName::CmpLtI32
            | KernelName::CmpLeI32
            | KernelName::CmpGtI32
            | KernelName::CmpGeI32
            | KernelName::BinaryAddI64
            | KernelName::BinarySubI64
            | KernelName::BinaryMulI64
            | KernelName::BinaryDivI64
            | KernelName::BinaryMaximumI64
            | KernelName::BinaryMinimumI64
            | KernelName::CmpEqI64
            | KernelName::CmpNeI64
            | KernelName::CmpLtI64
            | KernelName::CmpLeI64
            | KernelName::CmpGtI64
            | KernelName::CmpGeI64
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
            | KernelName::GemmF16
            | KernelName::GemmF64
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
            Self::GemmF16 => GEMM_F16_SPV,
            Self::GemmF64 => GEMM_F64_SPV,
            Self::ConstSet => CONST_SET_SPV,
            Self::UnarySlang => UNARY_SLANG_SPV,
            Self::BinarySlang => BINARY_SLANG_SPV,
            Self::CmpSlang => CMP_SLANG_SPV,
            Self::BinaryBf16 => BINARY_BF16_SPV,
            Self::CmpBf16 => CMP_BF16_SPV,
            Self::BinaryF8e4m3 => BINARY_F8E4M3_SPV,
            Self::CmpF8e4m3 => CMP_F8E4M3_SPV,
            Self::UnaryBf16 => UNARY_BF16_SPV,
            Self::UnaryF8e4m3 => UNARY_F8E4M3_SPV,
            Self::WhereSlang => WHERE_SLANG_SPV,
            Self::WhereBf16 => WHERE_BF16_SPV,
            Self::WhereF8e4m3 => WHERE_F8E4M3_SPV,
            Self::AffineBf16 => AFFINE_BF16_SPV,
            Self::AffineF8e4m3 => AFFINE_F8E4M3_SPV,
            Self::BinaryF16 => BINARY_F16_SPV,
            Self::CmpF16 => CMP_F16_SPV,
            Self::UnaryF16 => UNARY_F16_SPV,
            Self::AffineF16 => AFFINE_F16_SPV,
            Self::WhereF16 => WHERE_F16_SPV,
            Self::BinaryF64 => BINARY_F64_SPV,
            Self::CmpF64 => CMP_F64_SPV,
            Self::UnaryF64 => UNARY_F64_SPV,
            Self::AffineF64 => AFFINE_F64_SPV,
            Self::WhereF64 => WHERE_F64_SPV,
            Self::BinaryU8 => BINARY_U8_SPV,
            Self::CmpU8 => CMP_U8_SPV,
            Self::BinaryU32 => BINARY_U32_SPV,
            Self::CmpU32 => CMP_U32_SPV,
            Self::BinaryI16 => BINARY_I16_SPV,
            Self::CmpI16 => CMP_I16_SPV,
            Self::BinaryI32 => BINARY_I32_SPV,
            Self::CmpI32 => CMP_I32_SPV,
            Self::BinaryI64 => BINARY_I64_SPV,
            Self::CmpI64 => CMP_I64_SPV,
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
            Self::GemmF16 => "gemm_f16",
            Self::GemmF64 => "gemm_f64",
            Self::ConstSet => "const_set",
            Self::UnarySlang => "unary",
            Self::BinarySlang => "binary",
            Self::CmpSlang => "cmp",
            Self::BinaryBf16 => "binary_bf16",
            Self::CmpBf16 => "cmp_bf16",
            Self::BinaryF8e4m3 => "binary_f8e4m3",
            Self::CmpF8e4m3 => "cmp_f8e4m3",
            Self::UnaryBf16 => "unary_bf16",
            Self::UnaryF8e4m3 => "unary_f8e4m3",
            Self::WhereSlang => "where",
            Self::WhereBf16 => "where_bf16",
            Self::WhereF8e4m3 => "where_f8e4m3",
            Self::AffineBf16 => "affine_bf16",
            Self::AffineF8e4m3 => "affine_f8e4m3",
            Self::BinaryF16 => "binary_f16",
            Self::CmpF16 => "cmp_f16",
            Self::UnaryF16 => "unary_f16",
            Self::AffineF16 => "affine_f16",
            Self::WhereF16 => "where_f16",
            Self::BinaryF64 => "binary_f64",
            Self::CmpF64 => "cmp_f64",
            Self::UnaryF64 => "unary_f64",
            Self::AffineF64 => "affine_f64",
            Self::WhereF64 => "where_f64",
            Self::BinaryU8 => "binary_u8",
            Self::CmpU8 => "cmp_u8",
            Self::BinaryU32 => "binary_u32",
            Self::CmpU32 => "cmp_u32",
            Self::BinaryI16 => "binary_i16",
            Self::CmpI16 => "cmp_i16",
            Self::BinaryI32 => "binary_i32",
            Self::CmpI32 => "cmp_i32",
            Self::BinaryI64 => "binary_i64",
            Self::CmpI64 => "cmp_i64",
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
