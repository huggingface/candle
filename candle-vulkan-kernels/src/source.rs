//! Compiled SPIR-V binaries, generated from `shaders/*.slang` by `build.rs` via slangc.

use crate::kernel::KernelName;

const AFFINE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/affine.spv"));
const GEMM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm.spv"));
const GEMM_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_f16.spv"));
const GEMM_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_f64.spv"));
const CONST_SET_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/const_set.spv"));
const CONST_SET_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/const_set_f16.spv"));
const CONST_SET_BF16_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/const_set_bf16.spv"));
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
const COPY2D_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy2d_f16.spv"));
const COPY2D_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy2d_f64.spv"));
const COPY2D_BF16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy2d_bf16.spv"));
const COPY2D_U8_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy2d_u8.spv"));
const GATHER_IDX_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gather_idx.spv"));
const GATHER_IDX_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gather_idx_f16.spv"));
const GATHER_IDX_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gather_idx_f64.spv"));
const GATHER_IDX_BF16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gather_idx_bf16.spv"));
const GATHER_IDX_U8_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gather_idx_u8.spv"));
const REDUCE_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reduce_ops.spv"));
const REDUCE_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reduce_f16.spv"));
const REDUCE_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reduce_f64.spv"));
const REDUCE_BF16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reduce_bf16.spv"));
const POOL2D_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pool2d.spv"));
const POOL2D_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pool2d_f16.spv"));
const POOL2D_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pool2d_f64.spv"));
const UPSAMPLE_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/upsample.spv"));
const UPSAMPLE_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/upsample_f16.spv"));
const UPSAMPLE_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/upsample_f64.spv"));
const CONV_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv.spv"));
const CONV_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv_f16.spv"));
const CONV_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv_f64.spv"));
const CONV_BF16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv_bf16.spv"));
const SCATTER_SLANG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scatter.spv"));
const SCATTER_F16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scatter_f16.spv"));
const SCATTER_F64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scatter_f64.spv"));

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
const TO_DTYPE_F16F32_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f16_f32.spv"));
const TO_DTYPE_F16BF16_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f16_bf16.spv"));
const TO_DTYPE_F16F64_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f16_f64.spv"));
const TO_DTYPE_BF16F16_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_bf16_f16.spv"));
const TO_DTYPE_BF16F32_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_bf16_f32.spv"));
const TO_DTYPE_BF16F64_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_bf16_f64.spv"));
const TO_DTYPE_F32F16_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f32_f16.spv"));
const TO_DTYPE_F32BF16_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f32_bf16.spv"));
const TO_DTYPE_F32F64_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f32_f64.spv"));
const TO_DTYPE_F64F16_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f64_f16.spv"));
const TO_DTYPE_F64BF16_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f64_bf16.spv"));
const TO_DTYPE_F64F32_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f64_f32.spv"));
    static TO_DTYPE_U8U32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u8_u32.spv"));
    static TO_DTYPE_U8I16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u8_i16.spv"));
    static TO_DTYPE_U8I32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u8_i32.spv"));
    static TO_DTYPE_U8I64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u8_i64.spv"));
    static TO_DTYPE_U8BF16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u8_bf16.spv"));
    static TO_DTYPE_U8F16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u8_f16.spv"));
    static TO_DTYPE_U8F32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u8_f32.spv"));
    static TO_DTYPE_U8F64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u8_f64.spv"));
    static TO_DTYPE_U8F8E4M3_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u8_f8e4m3.spv"));
    static TO_DTYPE_U32U8_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u32_u8.spv"));
    static TO_DTYPE_U32I16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u32_i16.spv"));
    static TO_DTYPE_U32I32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u32_i32.spv"));
    static TO_DTYPE_U32I64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u32_i64.spv"));
    static TO_DTYPE_U32BF16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u32_bf16.spv"));
    static TO_DTYPE_U32F16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u32_f16.spv"));
    static TO_DTYPE_U32F32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u32_f32.spv"));
    static TO_DTYPE_U32F64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u32_f64.spv"));
    static TO_DTYPE_U32F8E4M3_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_u32_f8e4m3.spv"));
    static TO_DTYPE_I16U8_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i16_u8.spv"));
    static TO_DTYPE_I16U32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i16_u32.spv"));
    static TO_DTYPE_I16I32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i16_i32.spv"));
    static TO_DTYPE_I16I64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i16_i64.spv"));
    static TO_DTYPE_I16BF16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i16_bf16.spv"));
    static TO_DTYPE_I16F16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i16_f16.spv"));
    static TO_DTYPE_I16F32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i16_f32.spv"));
    static TO_DTYPE_I16F64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i16_f64.spv"));
    static TO_DTYPE_I16F8E4M3_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i16_f8e4m3.spv"));
    static TO_DTYPE_I32U8_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i32_u8.spv"));
    static TO_DTYPE_I32U32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i32_u32.spv"));
    static TO_DTYPE_I32I16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i32_i16.spv"));
    static TO_DTYPE_I32I64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i32_i64.spv"));
    static TO_DTYPE_I32BF16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i32_bf16.spv"));
    static TO_DTYPE_I32F16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i32_f16.spv"));
    static TO_DTYPE_I32F32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i32_f32.spv"));
    static TO_DTYPE_I32F64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i32_f64.spv"));
    static TO_DTYPE_I32F8E4M3_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i32_f8e4m3.spv"));
    static TO_DTYPE_I64U8_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i64_u8.spv"));
    static TO_DTYPE_I64U32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i64_u32.spv"));
    static TO_DTYPE_I64I16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i64_i16.spv"));
    static TO_DTYPE_I64I32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i64_i32.spv"));
    static TO_DTYPE_I64BF16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i64_bf16.spv"));
    static TO_DTYPE_I64F16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i64_f16.spv"));
    static TO_DTYPE_I64F32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i64_f32.spv"));
    static TO_DTYPE_I64F64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i64_f64.spv"));
    static TO_DTYPE_I64F8E4M3_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_i64_f8e4m3.spv"));
    static TO_DTYPE_F8E4M3U8_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f8e4m3_u8.spv"));
    static TO_DTYPE_F8E4M3U32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f8e4m3_u32.spv"));
    static TO_DTYPE_F8E4M3I16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f8e4m3_i16.spv"));
    static TO_DTYPE_F8E4M3I32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f8e4m3_i32.spv"));
    static TO_DTYPE_F8E4M3I64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f8e4m3_i64.spv"));
    static TO_DTYPE_F8E4M3BF16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f8e4m3_bf16.spv"));
    static TO_DTYPE_F8E4M3F16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f8e4m3_f16.spv"));
    static TO_DTYPE_F8E4M3F32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f8e4m3_f32.spv"));
    static TO_DTYPE_F8E4M3F64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f8e4m3_f64.spv"));
    static TO_DTYPE_BF16U8_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_bf16_u8.spv"));
    static TO_DTYPE_BF16U32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_bf16_u32.spv"));
    static TO_DTYPE_BF16I16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_bf16_i16.spv"));
    static TO_DTYPE_BF16I32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_bf16_i32.spv"));
    static TO_DTYPE_BF16I64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_bf16_i64.spv"));
    static TO_DTYPE_BF16F8E4M3_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_bf16_f8e4m3.spv"));
    static TO_DTYPE_F16U8_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f16_u8.spv"));
    static TO_DTYPE_F16U32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f16_u32.spv"));
    static TO_DTYPE_F16I16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f16_i16.spv"));
    static TO_DTYPE_F16I32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f16_i32.spv"));
    static TO_DTYPE_F16I64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f16_i64.spv"));
    static TO_DTYPE_F16F8E4M3_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f16_f8e4m3.spv"));
    static TO_DTYPE_F32U8_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f32_u8.spv"));
    static TO_DTYPE_F32U32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f32_u32.spv"));
    static TO_DTYPE_F32I16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f32_i16.spv"));
    static TO_DTYPE_F32I32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f32_i32.spv"));
    static TO_DTYPE_F32I64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f32_i64.spv"));
    static TO_DTYPE_F32F8E4M3_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f32_f8e4m3.spv"));
    static TO_DTYPE_F64U8_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f64_u8.spv"));
    static TO_DTYPE_F64U32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f64_u32.spv"));
    static TO_DTYPE_F64I16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f64_i16.spv"));
    static TO_DTYPE_F64I32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f64_i32.spv"));
    static TO_DTYPE_F64I64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f64_i64.spv"));
    static TO_DTYPE_F64F8E4M3_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/to_dtype_f64_f8e4m3.spv"));

    static INDEX_ADD_F32U32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/index_add_f32_u32.spv"));
    static INDEX_ADD_F32I64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/index_add_f32_i64.spv"));
    static INDEX_ADD_F32U8_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/index_add_f32_u8.spv"));
    static INDEX_ADD_F16U32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/index_add_f16_u32.spv"));
    static INDEX_ADD_F16I64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/index_add_f16_i64.spv"));
    static INDEX_ADD_F16U8_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/index_add_f16_u8.spv"));
    static INDEX_ADD_F64U32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/index_add_f64_u32.spv"));
    static INDEX_ADD_F64I64_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/index_add_f64_i64.spv"));
    static INDEX_ADD_F64U8_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/index_add_f64_u8.spv"));
    static DEQUANT_F32_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_f32.spv"));
    static DEQUANT_F16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_f16.spv"));
    static DEQUANT_BF16_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_bf16.spv"));
    static DEQUANT_Q4_0_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q4_0.spv"));
    static DEQUANT_Q4_1_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q4_1.spv"));
    static DEQUANT_Q5_0_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q5_0.spv"));
    static DEQUANT_Q5_1_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q5_1.spv"));
    static DEQUANT_Q8_0_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q8_0.spv"));
    static DEQUANT_Q8_1_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q8_1.spv"));
    static DEQUANT_Q2K_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q2k.spv"));
    static DEQUANT_Q3K_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q3k.spv"));
    static DEQUANT_Q4K_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q4k.spv"));
    static DEQUANT_Q5K_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q5k.spv"));
    static DEQUANT_Q6K_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q6k.spv"));
    static DEQUANT_Q8K_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q8k.spv"));
    static RMS_NORM_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm.spv"));
    static SOFTMAX_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/softmax.spv"));
    static ROPE_SPV: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv"));

/// The set of compiled compute shaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    Gemm,
    GemmF16,
    GemmF64,
    ConstSet,
    ConstSetF16,
    ConstSetBf16,
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
    Copy2dF16,
    Copy2dF64,
    Copy2dBf16,
    Copy2dU8,
    GatherIdxSlang,
    GatherIdxF16,
    GatherIdxF64,
    GatherIdxBf16,
    GatherIdxU8,
    ReduceSlang,
    ReduceF16,
    ReduceF64,
    ReduceBf16,
    Pool2dSlang,
    Pool2dF16,
    Pool2dF64,
    UpsampleSlang,
    UpsampleF16,
    UpsampleF64,
    ConvSlang,
    ConvF16,
    ConvF64,
    ConvBf16,
    ScatterSlang,
    ScatterF16,
    ScatterF64,
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
    ToDtypeF16F32,
    ToDtypeF16Bf16,
    ToDtypeF16F64,
    ToDtypeBf16F16,
    ToDtypeBf16F32,
    ToDtypeBf16F64,
    ToDtypeF32F16,
    ToDtypeF32Bf16,
    ToDtypeF32F64,
    ToDtypeF64F16,
    ToDtypeF64Bf16,
    ToDtypeF64F32,
    ToDtypeU8U32,
    ToDtypeU8I16,
    ToDtypeU8I32,
    ToDtypeU8I64,
    ToDtypeU8Bf16,
    ToDtypeU8F16,
    ToDtypeU8F32,
    ToDtypeU8F64,
    ToDtypeU8F8e4m3,
    ToDtypeU32U8,
    ToDtypeU32I16,
    ToDtypeU32I32,
    ToDtypeU32I64,
    ToDtypeU32Bf16,
    ToDtypeU32F16,
    ToDtypeU32F32,
    ToDtypeU32F64,
    ToDtypeU32F8e4m3,
    ToDtypeI16U8,
    ToDtypeI16U32,
    ToDtypeI16I32,
    ToDtypeI16I64,
    ToDtypeI16Bf16,
    ToDtypeI16F16,
    ToDtypeI16F32,
    ToDtypeI16F64,
    ToDtypeI16F8e4m3,
    ToDtypeI32U8,
    ToDtypeI32U32,
    ToDtypeI32I16,
    ToDtypeI32I64,
    ToDtypeI32Bf16,
    ToDtypeI32F16,
    ToDtypeI32F32,
    ToDtypeI32F64,
    ToDtypeI32F8e4m3,
    ToDtypeI64U8,
    ToDtypeI64U32,
    ToDtypeI64I16,
    ToDtypeI64I32,
    ToDtypeI64Bf16,
    ToDtypeI64F16,
    ToDtypeI64F32,
    ToDtypeI64F64,
    ToDtypeI64F8e4m3,
    ToDtypeF8e4m3U8,
    ToDtypeF8e4m3U32,
    ToDtypeF8e4m3I16,
    ToDtypeF8e4m3I32,
    ToDtypeF8e4m3I64,
    ToDtypeF8e4m3Bf16,
    ToDtypeF8e4m3F16,
    ToDtypeF8e4m3F32,
    ToDtypeF8e4m3F64,
    ToDtypeBf16U8,
    ToDtypeBf16U32,
    ToDtypeBf16I16,
    ToDtypeBf16I32,
    ToDtypeBf16I64,
    ToDtypeBf16F8e4m3,
    ToDtypeF16U8,
    ToDtypeF16U32,
    ToDtypeF16I16,
    ToDtypeF16I32,
    ToDtypeF16I64,
    ToDtypeF16F8e4m3,
    ToDtypeF32U8,
    ToDtypeF32U32,
    ToDtypeF32I16,
    ToDtypeF32I32,
    ToDtypeF32I64,
    ToDtypeF32F8e4m3,
    ToDtypeF64U8,
    ToDtypeF64U32,
    ToDtypeF64I16,
    ToDtypeF64I32,
    ToDtypeF64I64,
    ToDtypeF64F8e4m3,
    IndexAddF32U32,
    IndexAddF32I64,
    IndexAddF32U8,
    IndexAddF16U32,
    IndexAddF16I64,
    IndexAddF16U8,
    IndexAddF64U32,
    IndexAddF64I64,
    IndexAddF64U8,
    DequantF32,
    DequantF16,
    DequantBf16,
    DequantQ40,
    DequantQ41,
    DequantQ50,
    DequantQ51,
    DequantQ80,
    DequantQ81,
    DequantQ2K,
    DequantQ3K,
    DequantQ4K,
    DequantQ5K,
    DequantQ6K,
    DequantQ8K,
    RmsNorm,
    Softmax,
    Rope,
}

impl Source {
    /// Size of the push constant block for this shader entry point, in
    /// bytes. The pipeline layout must declare exactly the bytes the shader
    /// accesses, otherwise dispatch fails validation.
    pub fn push_constant_size(self, name: KernelName) -> u32 {
        match name {
            KernelName::ConstSetF32 | KernelName::ConstSetF16 | KernelName::ConstSetBf16 => 0,
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
            | KernelName::Copy2dF16
            | KernelName::Copy2dF64
            | KernelName::Copy2dBf16
            | KernelName::Copy2dU8
            | KernelName::GatherIdxF32
            | KernelName::GatherRowsF32
            | KernelName::IndexSelectF32
            | KernelName::GatherIdxF16
            | KernelName::GatherRowsF16
            | KernelName::IndexSelectF16
            | KernelName::GatherIdxF64
            | KernelName::GatherIdxBf16
            | KernelName::GatherIdxU8
            | KernelName::GatherRowsF64
            | KernelName::IndexSelectF64
            | KernelName::ReduceMinF32
            | KernelName::ReduceArgMinF32
            | KernelName::ReduceArgMaxF32
            | KernelName::ReduceSumF32
            | KernelName::ReduceMaxF32
            | KernelName::ReduceSumF16
            | KernelName::ReduceMaxF16
            | KernelName::ReduceMinF16
            | KernelName::ReduceArgMinF16
            | KernelName::ReduceArgMaxF16
            | KernelName::ReduceSumF64
            | KernelName::ReduceMaxF64
            | KernelName::ReduceMinF64
            | KernelName::ReduceArgMinF64
            | KernelName::ReduceArgMaxF64
            | KernelName::ReduceSumBf16
            | KernelName::ReduceMaxBf16
            | KernelName::ReduceMinBf16
            | KernelName::ReduceArgMinBf16
            | KernelName::ReduceArgMaxBf16
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
            | KernelName::AvgPool2dF16
            | KernelName::MaxPool2dF16
            | KernelName::UpsampleNearest1dF16
            | KernelName::UpsampleNearest2dF16
            | KernelName::UpsampleBilinear2dF16
            | KernelName::AvgPool2dF64
            | KernelName::MaxPool2dF64
            | KernelName::UpsampleNearest1dF64
            | KernelName::UpsampleNearest2dF64
            | KernelName::UpsampleBilinear2dF64
            | KernelName::Conv1dF32
            | KernelName::Conv2dF32
            | KernelName::ConvTranspose1dF32
            | KernelName::ConvTranspose2dF32
            | KernelName::Conv1dF16
            | KernelName::Conv2dF16
            | KernelName::ConvTranspose1dF16
            | KernelName::ConvTranspose2dF16
            | KernelName::Conv1dF64
            | KernelName::Conv2dF64
            | KernelName::ConvTranspose1dF64
            | KernelName::ConvTranspose2dF64
            | KernelName::ConvTranspose2dBf16
            | KernelName::ScatterF32
            | KernelName::ScatterF16
            | KernelName::ScatterF64 => 0,
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
            KernelName::ToDtypeF16F32
            | KernelName::ToDtypeF16Bf16
            | KernelName::ToDtypeF16F64
            | KernelName::ToDtypeBf16F16
            | KernelName::ToDtypeBf16F32
            | KernelName::ToDtypeBf16F64
            | KernelName::ToDtypeF32F16
            | KernelName::ToDtypeF32Bf16
            | KernelName::ToDtypeF32F64
            | KernelName::ToDtypeF64F16
            | KernelName::ToDtypeF64Bf16
            | KernelName::ToDtypeF64F32 
            | KernelName::ToDtypeU8U32
            | KernelName::ToDtypeU8I16
            | KernelName::ToDtypeU8I32
            | KernelName::ToDtypeU8I64
            | KernelName::ToDtypeU8Bf16
            | KernelName::ToDtypeU8F16
            | KernelName::ToDtypeU8F32
            | KernelName::ToDtypeU8F64
            | KernelName::ToDtypeU8F8e4m3
            | KernelName::ToDtypeU32U8
            | KernelName::ToDtypeU32I16
            | KernelName::ToDtypeU32I32
            | KernelName::ToDtypeU32I64
            | KernelName::ToDtypeU32Bf16
            | KernelName::ToDtypeU32F16
            | KernelName::ToDtypeU32F32
            | KernelName::ToDtypeU32F64
            | KernelName::ToDtypeU32F8e4m3
            | KernelName::ToDtypeI16U8
            | KernelName::ToDtypeI16U32
            | KernelName::ToDtypeI16I32
            | KernelName::ToDtypeI16I64
            | KernelName::ToDtypeI16Bf16
            | KernelName::ToDtypeI16F16
            | KernelName::ToDtypeI16F32
            | KernelName::ToDtypeI16F64
            | KernelName::ToDtypeI16F8e4m3
            | KernelName::ToDtypeI32U8
            | KernelName::ToDtypeI32U32
            | KernelName::ToDtypeI32I16
            | KernelName::ToDtypeI32I64
            | KernelName::ToDtypeI32Bf16
            | KernelName::ToDtypeI32F16
            | KernelName::ToDtypeI32F32
            | KernelName::ToDtypeI32F64
            | KernelName::ToDtypeI32F8e4m3
            | KernelName::ToDtypeI64U8
            | KernelName::ToDtypeI64U32
            | KernelName::ToDtypeI64I16
            | KernelName::ToDtypeI64I32
            | KernelName::ToDtypeI64Bf16
            | KernelName::ToDtypeI64F16
            | KernelName::ToDtypeI64F32
            | KernelName::ToDtypeI64F64
            | KernelName::ToDtypeI64F8e4m3
            | KernelName::ToDtypeF8e4m3U8
            | KernelName::ToDtypeF8e4m3U32
            | KernelName::ToDtypeF8e4m3I16
            | KernelName::ToDtypeF8e4m3I32
            | KernelName::ToDtypeF8e4m3I64
            | KernelName::ToDtypeF8e4m3Bf16
            | KernelName::ToDtypeF8e4m3F16
            | KernelName::ToDtypeF8e4m3F32
            | KernelName::ToDtypeF8e4m3F64
            | KernelName::ToDtypeBf16U8
            | KernelName::ToDtypeBf16U32
            | KernelName::ToDtypeBf16I16
            | KernelName::ToDtypeBf16I32
            | KernelName::ToDtypeBf16I64
            | KernelName::ToDtypeBf16F8e4m3
            | KernelName::ToDtypeF16U8
            | KernelName::ToDtypeF16U32
            | KernelName::ToDtypeF16I16
            | KernelName::ToDtypeF16I32
            | KernelName::ToDtypeF16I64
            | KernelName::ToDtypeF16F8e4m3
            | KernelName::ToDtypeF32U8
            | KernelName::ToDtypeF32U32
            | KernelName::ToDtypeF32I16
            | KernelName::ToDtypeF32I32
            | KernelName::ToDtypeF32I64
            | KernelName::ToDtypeF32F8e4m3
            | KernelName::ToDtypeF64U8
            | KernelName::ToDtypeF64U32
            | KernelName::ToDtypeF64I16
            | KernelName::ToDtypeF64I32
            | KernelName::ToDtypeF64I64
            | KernelName::ToDtypeF64F8e4m3
            | KernelName::IndexAddF32U32
            | KernelName::IndexAddF32I64
            | KernelName::IndexAddF32U8
            | KernelName::IndexAddF16U32
            | KernelName::IndexAddF16I64
            | KernelName::IndexAddF16U8
            | KernelName::IndexAddF64U32
            | KernelName::IndexAddF64I64
            | KernelName::IndexAddF64U8
            | KernelName::DequantF32
            | KernelName::DequantF16
            | KernelName::DequantBf16
            | KernelName::DequantQ40
            | KernelName::DequantQ41
            | KernelName::DequantQ50
            | KernelName::DequantQ51
            | KernelName::DequantQ80
            | KernelName::DequantQ81
            | KernelName::DequantQ2K
            | KernelName::DequantQ3K
            | KernelName::DequantQ4K
            | KernelName::DequantQ5K
            | KernelName::DequantQ6K
            | KernelName::DequantQ8K
            | KernelName::RmsNorm
            | KernelName::Softmax
            | KernelName::Rope
            => 0,
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
            Self::ConstSetF16 => CONST_SET_F16_SPV,
            Self::ConstSetBf16 => CONST_SET_BF16_SPV,
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
            Self::Copy2dF16 => COPY2D_F16_SPV,
            Self::Copy2dF64 => COPY2D_F64_SPV,
            Self::Copy2dBf16 => COPY2D_BF16_SPV,
            Self::Copy2dU8 => COPY2D_U8_SPV,
            Self::GatherIdxSlang => GATHER_IDX_SLANG_SPV,
            Self::GatherIdxF16 => GATHER_IDX_F16_SPV,
            Self::GatherIdxF64 => GATHER_IDX_F64_SPV,
            Self::GatherIdxBf16 => GATHER_IDX_BF16_SPV,
            Self::GatherIdxU8 => GATHER_IDX_U8_SPV,
            Self::ReduceSlang => REDUCE_SLANG_SPV,
            Self::ReduceF16 => REDUCE_F16_SPV,
            Self::ReduceF64 => REDUCE_F64_SPV,
            Self::ReduceBf16 => REDUCE_BF16_SPV,
            Self::Pool2dSlang => POOL2D_SLANG_SPV,
            Self::Pool2dF16 => POOL2D_F16_SPV,
            Self::Pool2dF64 => POOL2D_F64_SPV,
            Self::UpsampleSlang => UPSAMPLE_SLANG_SPV,
            Self::UpsampleF16 => UPSAMPLE_F16_SPV,
            Self::UpsampleF64 => UPSAMPLE_F64_SPV,
            Self::ConvSlang => CONV_SLANG_SPV,
            Self::ConvF16 => CONV_F16_SPV,
            Self::ConvF64 => CONV_F64_SPV,
            Self::ConvBf16 => CONV_BF16_SPV,
            Self::ScatterSlang => SCATTER_SLANG_SPV,
            Self::ScatterF16 => SCATTER_F16_SPV,
            Self::ScatterF64 => SCATTER_F64_SPV,
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
            Self::ToDtypeF16F32 => TO_DTYPE_F16F32_SPV,
            Self::ToDtypeF16Bf16 => TO_DTYPE_F16BF16_SPV,
            Self::ToDtypeF16F64 => TO_DTYPE_F16F64_SPV,
            Self::ToDtypeBf16F16 => TO_DTYPE_BF16F16_SPV,
            Self::ToDtypeBf16F32 => TO_DTYPE_BF16F32_SPV,
            Self::ToDtypeBf16F64 => TO_DTYPE_BF16F64_SPV,
            Self::ToDtypeF32F16 => TO_DTYPE_F32F16_SPV,
            Self::ToDtypeF32Bf16 => TO_DTYPE_F32BF16_SPV,
            Self::ToDtypeF32F64 => TO_DTYPE_F32F64_SPV,
            Self::ToDtypeF64F16 => TO_DTYPE_F64F16_SPV,
            Self::ToDtypeF64Bf16 => TO_DTYPE_F64BF16_SPV,
            Self::ToDtypeF64F32 => TO_DTYPE_F64F32_SPV,
            Self::ToDtypeU8U32 => TO_DTYPE_U8U32_SPV,
            Self::ToDtypeU8I16 => TO_DTYPE_U8I16_SPV,
            Self::ToDtypeU8I32 => TO_DTYPE_U8I32_SPV,
            Self::ToDtypeU8I64 => TO_DTYPE_U8I64_SPV,
            Self::ToDtypeU8Bf16 => TO_DTYPE_U8BF16_SPV,
            Self::ToDtypeU8F16 => TO_DTYPE_U8F16_SPV,
            Self::ToDtypeU8F32 => TO_DTYPE_U8F32_SPV,
            Self::ToDtypeU8F64 => TO_DTYPE_U8F64_SPV,
            Self::ToDtypeU8F8e4m3 => TO_DTYPE_U8F8E4M3_SPV,
            Self::ToDtypeU32U8 => TO_DTYPE_U32U8_SPV,
            Self::ToDtypeU32I16 => TO_DTYPE_U32I16_SPV,
            Self::ToDtypeU32I32 => TO_DTYPE_U32I32_SPV,
            Self::ToDtypeU32I64 => TO_DTYPE_U32I64_SPV,
            Self::ToDtypeU32Bf16 => TO_DTYPE_U32BF16_SPV,
            Self::ToDtypeU32F16 => TO_DTYPE_U32F16_SPV,
            Self::ToDtypeU32F32 => TO_DTYPE_U32F32_SPV,
            Self::ToDtypeU32F64 => TO_DTYPE_U32F64_SPV,
            Self::ToDtypeU32F8e4m3 => TO_DTYPE_U32F8E4M3_SPV,
            Self::ToDtypeI16U8 => TO_DTYPE_I16U8_SPV,
            Self::ToDtypeI16U32 => TO_DTYPE_I16U32_SPV,
            Self::ToDtypeI16I32 => TO_DTYPE_I16I32_SPV,
            Self::ToDtypeI16I64 => TO_DTYPE_I16I64_SPV,
            Self::ToDtypeI16Bf16 => TO_DTYPE_I16BF16_SPV,
            Self::ToDtypeI16F16 => TO_DTYPE_I16F16_SPV,
            Self::ToDtypeI16F32 => TO_DTYPE_I16F32_SPV,
            Self::ToDtypeI16F64 => TO_DTYPE_I16F64_SPV,
            Self::ToDtypeI16F8e4m3 => TO_DTYPE_I16F8E4M3_SPV,
            Self::ToDtypeI32U8 => TO_DTYPE_I32U8_SPV,
            Self::ToDtypeI32U32 => TO_DTYPE_I32U32_SPV,
            Self::ToDtypeI32I16 => TO_DTYPE_I32I16_SPV,
            Self::ToDtypeI32I64 => TO_DTYPE_I32I64_SPV,
            Self::ToDtypeI32Bf16 => TO_DTYPE_I32BF16_SPV,
            Self::ToDtypeI32F16 => TO_DTYPE_I32F16_SPV,
            Self::ToDtypeI32F32 => TO_DTYPE_I32F32_SPV,
            Self::ToDtypeI32F64 => TO_DTYPE_I32F64_SPV,
            Self::ToDtypeI32F8e4m3 => TO_DTYPE_I32F8E4M3_SPV,
            Self::ToDtypeI64U8 => TO_DTYPE_I64U8_SPV,
            Self::ToDtypeI64U32 => TO_DTYPE_I64U32_SPV,
            Self::ToDtypeI64I16 => TO_DTYPE_I64I16_SPV,
            Self::ToDtypeI64I32 => TO_DTYPE_I64I32_SPV,
            Self::ToDtypeI64Bf16 => TO_DTYPE_I64BF16_SPV,
            Self::ToDtypeI64F16 => TO_DTYPE_I64F16_SPV,
            Self::ToDtypeI64F32 => TO_DTYPE_I64F32_SPV,
            Self::ToDtypeI64F64 => TO_DTYPE_I64F64_SPV,
            Self::ToDtypeI64F8e4m3 => TO_DTYPE_I64F8E4M3_SPV,
            Self::ToDtypeF8e4m3U8 => TO_DTYPE_F8E4M3U8_SPV,
            Self::ToDtypeF8e4m3U32 => TO_DTYPE_F8E4M3U32_SPV,
            Self::ToDtypeF8e4m3I16 => TO_DTYPE_F8E4M3I16_SPV,
            Self::ToDtypeF8e4m3I32 => TO_DTYPE_F8E4M3I32_SPV,
            Self::ToDtypeF8e4m3I64 => TO_DTYPE_F8E4M3I64_SPV,
            Self::ToDtypeF8e4m3Bf16 => TO_DTYPE_F8E4M3BF16_SPV,
            Self::ToDtypeF8e4m3F16 => TO_DTYPE_F8E4M3F16_SPV,
            Self::ToDtypeF8e4m3F32 => TO_DTYPE_F8E4M3F32_SPV,
            Self::ToDtypeF8e4m3F64 => TO_DTYPE_F8E4M3F64_SPV,
            Self::ToDtypeBf16U8 => TO_DTYPE_BF16U8_SPV,
            Self::ToDtypeBf16U32 => TO_DTYPE_BF16U32_SPV,
            Self::ToDtypeBf16I16 => TO_DTYPE_BF16I16_SPV,
            Self::ToDtypeBf16I32 => TO_DTYPE_BF16I32_SPV,
            Self::ToDtypeBf16I64 => TO_DTYPE_BF16I64_SPV,
            Self::ToDtypeBf16F8e4m3 => TO_DTYPE_BF16F8E4M3_SPV,
            Self::ToDtypeF16U8 => TO_DTYPE_F16U8_SPV,
            Self::ToDtypeF16U32 => TO_DTYPE_F16U32_SPV,
            Self::ToDtypeF16I16 => TO_DTYPE_F16I16_SPV,
            Self::ToDtypeF16I32 => TO_DTYPE_F16I32_SPV,
            Self::ToDtypeF16I64 => TO_DTYPE_F16I64_SPV,
            Self::ToDtypeF16F8e4m3 => TO_DTYPE_F16F8E4M3_SPV,
            Self::ToDtypeF32U8 => TO_DTYPE_F32U8_SPV,
            Self::ToDtypeF32U32 => TO_DTYPE_F32U32_SPV,
            Self::ToDtypeF32I16 => TO_DTYPE_F32I16_SPV,
            Self::ToDtypeF32I32 => TO_DTYPE_F32I32_SPV,
            Self::ToDtypeF32I64 => TO_DTYPE_F32I64_SPV,
            Self::ToDtypeF32F8e4m3 => TO_DTYPE_F32F8E4M3_SPV,
            Self::ToDtypeF64U8 => TO_DTYPE_F64U8_SPV,
            Self::ToDtypeF64U32 => TO_DTYPE_F64U32_SPV,
            Self::ToDtypeF64I16 => TO_DTYPE_F64I16_SPV,
            Self::ToDtypeF64I32 => TO_DTYPE_F64I32_SPV,
            Self::ToDtypeF64I64 => TO_DTYPE_F64I64_SPV,
            Self::ToDtypeF64F8e4m3 => TO_DTYPE_F64F8E4M3_SPV,
            Self::IndexAddF32U32 => INDEX_ADD_F32U32_SPV,
            Self::IndexAddF32I64 => INDEX_ADD_F32I64_SPV,
            Self::IndexAddF32U8 => INDEX_ADD_F32U8_SPV,
            Self::IndexAddF16U32 => INDEX_ADD_F16U32_SPV,
            Self::IndexAddF16I64 => INDEX_ADD_F16I64_SPV,
            Self::IndexAddF16U8 => INDEX_ADD_F16U8_SPV,
            Self::IndexAddF64U32 => INDEX_ADD_F64U32_SPV,
            Self::IndexAddF64I64 => INDEX_ADD_F64I64_SPV,
            Self::IndexAddF64U8 => INDEX_ADD_F64U8_SPV,
            Self::DequantF32 => DEQUANT_F32_SPV,
            Self::DequantF16 => DEQUANT_F16_SPV,
            Self::DequantBf16 => DEQUANT_BF16_SPV,
            Self::DequantQ40 => DEQUANT_Q4_0_SPV,
            Self::DequantQ41 => DEQUANT_Q4_1_SPV,
            Self::DequantQ50 => DEQUANT_Q5_0_SPV,
            Self::DequantQ51 => DEQUANT_Q5_1_SPV,
            Self::DequantQ80 => DEQUANT_Q8_0_SPV,
            Self::DequantQ81 => DEQUANT_Q8_1_SPV,
            Self::DequantQ2K => DEQUANT_Q2K_SPV,
            Self::DequantQ3K => DEQUANT_Q3K_SPV,
            Self::DequantQ4K => DEQUANT_Q4K_SPV,
            Self::DequantQ5K => DEQUANT_Q5K_SPV,
            Self::DequantQ6K => DEQUANT_Q6K_SPV,
            Self::DequantQ8K => DEQUANT_Q8K_SPV,
            Self::RmsNorm => RMS_NORM_SPV,
            Self::Softmax => SOFTMAX_SPV,
            Self::Rope => ROPE_SPV,
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
            Self::ConstSetF16 => "const_set_f16",
            Self::ConstSetBf16 => "const_set_bf16",
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
            Self::Copy2dF16 => "copy2d_f16",
            Self::Copy2dF64 => "copy2d_f64",
            Self::Copy2dBf16 => "copy2d_bf16",
            Self::Copy2dU8 => "copy2d_u8",
            Self::GatherIdxSlang => "gather_idx",
            Self::GatherIdxF16 => "gather_idx_f16",
            Self::GatherIdxF64 => "gather_idx_f64",
            Self::GatherIdxBf16 => "gather_idx_bf16",
            Self::GatherIdxU8 => "gather_idx_u8",
            Self::ReduceSlang => "reduce",
            Self::ReduceF16 => "reduce_f16",
            Self::ReduceF64 => "reduce_f64",
            Self::ReduceBf16 => "reduce_bf16",
            Self::Pool2dSlang => "pool2d",
            Self::Pool2dF16 => "pool2d_f16",
            Self::Pool2dF64 => "pool2d_f64",
            Self::UpsampleSlang => "upsample",
            Self::UpsampleF16 => "upsample_f16",
            Self::UpsampleF64 => "upsample_f64",
            Self::ConvSlang => "conv",
            Self::ConvF16 => "conv_f16",
            Self::ConvF64 => "conv_f64",
            Self::ConvBf16 => "conv_bf16",
            Self::ScatterSlang => "scatter",
            Self::ScatterF16 => "scatter_f16",
            Self::ScatterF64 => "scatter_f64",
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
            Self::ToDtypeF16F32 => "to_dtype_f16_f32",
            Self::ToDtypeF16Bf16 => "to_dtype_f16_bf16",
            Self::ToDtypeF16F64 => "to_dtype_f16_f64",
            Self::ToDtypeBf16F16 => "to_dtype_bf16_f16",
            Self::ToDtypeBf16F32 => "to_dtype_bf16_f32",
            Self::ToDtypeBf16F64 => "to_dtype_bf16_f64",
            Self::ToDtypeF32F16 => "to_dtype_f32_f16",
            Self::ToDtypeF32Bf16 => "to_dtype_f32_bf16",
            Self::ToDtypeF32F64 => "to_dtype_f32_f64",
            Self::ToDtypeF64F16 => "to_dtype_f64_f16",
            Self::ToDtypeF64Bf16 => "to_dtype_f64_bf16",
            Self::ToDtypeF64F32 => "to_dtype_f64_f32",
            Self::ToDtypeU8U32 => "to_dtype_u8_u32",
            Self::ToDtypeU8I16 => "to_dtype_u8_i16",
            Self::ToDtypeU8I32 => "to_dtype_u8_i32",
            Self::ToDtypeU8I64 => "to_dtype_u8_i64",
            Self::ToDtypeU8Bf16 => "to_dtype_u8_bf16",
            Self::ToDtypeU8F16 => "to_dtype_u8_f16",
            Self::ToDtypeU8F32 => "to_dtype_u8_f32",
            Self::ToDtypeU8F64 => "to_dtype_u8_f64",
            Self::ToDtypeU8F8e4m3 => "to_dtype_u8_f8e4m3",
            Self::ToDtypeU32U8 => "to_dtype_u32_u8",
            Self::ToDtypeU32I16 => "to_dtype_u32_i16",
            Self::ToDtypeU32I32 => "to_dtype_u32_i32",
            Self::ToDtypeU32I64 => "to_dtype_u32_i64",
            Self::ToDtypeU32Bf16 => "to_dtype_u32_bf16",
            Self::ToDtypeU32F16 => "to_dtype_u32_f16",
            Self::ToDtypeU32F32 => "to_dtype_u32_f32",
            Self::ToDtypeU32F64 => "to_dtype_u32_f64",
            Self::ToDtypeU32F8e4m3 => "to_dtype_u32_f8e4m3",
            Self::ToDtypeI16U8 => "to_dtype_i16_u8",
            Self::ToDtypeI16U32 => "to_dtype_i16_u32",
            Self::ToDtypeI16I32 => "to_dtype_i16_i32",
            Self::ToDtypeI16I64 => "to_dtype_i16_i64",
            Self::ToDtypeI16Bf16 => "to_dtype_i16_bf16",
            Self::ToDtypeI16F16 => "to_dtype_i16_f16",
            Self::ToDtypeI16F32 => "to_dtype_i16_f32",
            Self::ToDtypeI16F64 => "to_dtype_i16_f64",
            Self::ToDtypeI16F8e4m3 => "to_dtype_i16_f8e4m3",
            Self::ToDtypeI32U8 => "to_dtype_i32_u8",
            Self::ToDtypeI32U32 => "to_dtype_i32_u32",
            Self::ToDtypeI32I16 => "to_dtype_i32_i16",
            Self::ToDtypeI32I64 => "to_dtype_i32_i64",
            Self::ToDtypeI32Bf16 => "to_dtype_i32_bf16",
            Self::ToDtypeI32F16 => "to_dtype_i32_f16",
            Self::ToDtypeI32F32 => "to_dtype_i32_f32",
            Self::ToDtypeI32F64 => "to_dtype_i32_f64",
            Self::ToDtypeI32F8e4m3 => "to_dtype_i32_f8e4m3",
            Self::ToDtypeI64U8 => "to_dtype_i64_u8",
            Self::ToDtypeI64U32 => "to_dtype_i64_u32",
            Self::ToDtypeI64I16 => "to_dtype_i64_i16",
            Self::ToDtypeI64I32 => "to_dtype_i64_i32",
            Self::ToDtypeI64Bf16 => "to_dtype_i64_bf16",
            Self::ToDtypeI64F16 => "to_dtype_i64_f16",
            Self::ToDtypeI64F32 => "to_dtype_i64_f32",
            Self::ToDtypeI64F64 => "to_dtype_i64_f64",
            Self::ToDtypeI64F8e4m3 => "to_dtype_i64_f8e4m3",
            Self::ToDtypeF8e4m3U8 => "to_dtype_f8e4m3_u8",
            Self::ToDtypeF8e4m3U32 => "to_dtype_f8e4m3_u32",
            Self::ToDtypeF8e4m3I16 => "to_dtype_f8e4m3_i16",
            Self::ToDtypeF8e4m3I32 => "to_dtype_f8e4m3_i32",
            Self::ToDtypeF8e4m3I64 => "to_dtype_f8e4m3_i64",
            Self::ToDtypeF8e4m3Bf16 => "to_dtype_f8e4m3_bf16",
            Self::ToDtypeF8e4m3F16 => "to_dtype_f8e4m3_f16",
            Self::ToDtypeF8e4m3F32 => "to_dtype_f8e4m3_f32",
            Self::ToDtypeF8e4m3F64 => "to_dtype_f8e4m3_f64",
            Self::ToDtypeBf16U8 => "to_dtype_bf16_u8",
            Self::ToDtypeBf16U32 => "to_dtype_bf16_u32",
            Self::ToDtypeBf16I16 => "to_dtype_bf16_i16",
            Self::ToDtypeBf16I32 => "to_dtype_bf16_i32",
            Self::ToDtypeBf16I64 => "to_dtype_bf16_i64",
            Self::ToDtypeBf16F8e4m3 => "to_dtype_bf16_f8e4m3",
            Self::ToDtypeF16U8 => "to_dtype_f16_u8",
            Self::ToDtypeF16U32 => "to_dtype_f16_u32",
            Self::ToDtypeF16I16 => "to_dtype_f16_i16",
            Self::ToDtypeF16I32 => "to_dtype_f16_i32",
            Self::ToDtypeF16I64 => "to_dtype_f16_i64",
            Self::ToDtypeF16F8e4m3 => "to_dtype_f16_f8e4m3",
            Self::ToDtypeF32U8 => "to_dtype_f32_u8",
            Self::ToDtypeF32U32 => "to_dtype_f32_u32",
            Self::ToDtypeF32I16 => "to_dtype_f32_i16",
            Self::ToDtypeF32I32 => "to_dtype_f32_i32",
            Self::ToDtypeF32I64 => "to_dtype_f32_i64",
            Self::ToDtypeF32F8e4m3 => "to_dtype_f32_f8e4m3",
            Self::ToDtypeF64U8 => "to_dtype_f64_u8",
            Self::ToDtypeF64U32 => "to_dtype_f64_u32",
            Self::ToDtypeF64I16 => "to_dtype_f64_i16",
            Self::ToDtypeF64I32 => "to_dtype_f64_i32",
            Self::ToDtypeF64I64 => "to_dtype_f64_i64",
            Self::ToDtypeF64F8e4m3 => "to_dtype_f64_f8e4m3",
            Self::IndexAddF32U32 => "index_add_f32_u32",
            Self::IndexAddF32I64 => "index_add_f32_i64",
            Self::IndexAddF32U8 => "index_add_f32_u8",
            Self::IndexAddF16U32 => "index_add_f16_u32",
            Self::IndexAddF16I64 => "index_add_f16_i64",
            Self::IndexAddF16U8 => "index_add_f16_u8",
            Self::IndexAddF64U32 => "index_add_f64_u32",
            Self::IndexAddF64I64 => "index_add_f64_i64",
            Self::IndexAddF64U8 => "index_add_f64_u8",
            Self::DequantF32 => "dequant_f32",
            Self::DequantF16 => "dequant_f16",
            Self::DequantBf16 => "dequant_bf16",
            Self::DequantQ40 => "dequant_q4_0",
            Self::DequantQ41 => "dequant_q4_1",
            Self::DequantQ50 => "dequant_q5_0",
            Self::DequantQ51 => "dequant_q5_1",
            Self::DequantQ80 => "dequant_q8_0",
            Self::DequantQ81 => "dequant_q8_1",
            Self::DequantQ2K => "dequant_q2k",
            Self::DequantQ3K => "dequant_q3k",
            Self::DequantQ4K => "dequant_q4k",
            Self::DequantQ5K => "dequant_q5k",
            Self::DequantQ6K => "dequant_q6k",
            Self::DequantQ8K => "dequant_q8k",
            Self::RmsNorm => "rms_norm",
            Self::Softmax => "softmax",
            Self::Rope => "rope",
        }
    }
}
