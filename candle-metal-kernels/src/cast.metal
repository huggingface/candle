#include <metal_stdlib>

METAL_FUNC uint get_strided_index(
    uint idx,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}


using namespace metal;

#define CAST(FN_NAME, FN_NAME_STRIDED, LEFT_TYPENAME, RIGHT_TYPENAME) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const LEFT_TYPENAME *input,  \
    device RIGHT_TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = static_cast<RIGHT_TYPENAME>(input[tid]); \
} \
kernel void FN_NAME_STRIDED( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    device const LEFT_TYPENAME *input,  \
    device RIGHT_TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = static_cast<RIGHT_TYPENAME>(input[get_strided_index(tid, num_dims, dims, strides)]); \
} \

#define CAST_THROUGH(FN_NAME, FN_NAME_STRIDED, LEFT_TYPENAME, RIGHT_TYPENAME, IR_TYPENAME) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const LEFT_TYPENAME *input,  \
    device RIGHT_TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = static_cast<RIGHT_TYPENAME>(static_cast<IR_TYPENAME>(input[tid])); \
} \
kernel void FN_NAME_STRIDED( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    device const LEFT_TYPENAME *input,  \
    device RIGHT_TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = static_cast<RIGHT_TYPENAME>(static_cast<IR_TYPENAME>(input[get_strided_index(tid, num_dims, dims, strides)])); \
} \

uint8_t f32_to_fp8e4m3(float x) {
    // Transmute float to uint32
    uint32_t xbits = as_type<uint32_t>(x);

    // Constants for E4M3 interpretation
    const uint8_t fp8_mantissa_mask = 0x7;
    const uint16_t fp8_exp_bias = 7;
    const uint32_t fp8_significand_bits = 4;
    const uint32_t fp8_mindenorm_o2 = 0x38800000;  // 2^-15 / 2 as float bits
    const uint32_t fp8_overflow_threshold = 0x43000000;  // 2^8 as float bits
    const uint32_t fp8_minnorm = 0x39000000;  // 2^-14 as float bits
    const uint32_t FP_INF_BITS = 0x7F800000;  // Infinity for float

    // Half ULP for FP8 rounding
    const uint32_t fp8_fp_half_ulp = 1 << (23 - fp8_significand_bits - 1);

    // Extract components
    uint8_t sign = ((xbits >> 31) << 7);
    uint8_t exp = (((((xbits >> 23) & 0xFF) - 127) + fp8_exp_bias) & 0xFF);
    uint8_t mantissa = (xbits >> (23 - fp8_significand_bits)) & fp8_mantissa_mask;
    uint32_t absx = xbits & 0x7FFFFFFF;

    uint8_t res;

    if (absx <= fp8_mindenorm_o2) {
        // Zero or underflow
        res = 0;
    } else if (absx > FP_INF_BITS) {
        // Preserve NaNs
        res = 0x7F;
    } else if (absx > fp8_overflow_threshold) {
        // Saturate (NoSat -> NaN)
        res = 0x7F; // NaN for NoSat
    } else if (absx >= fp8_minnorm) {
        // Normal range
        res = (exp << (fp8_significand_bits - 1)) | mantissa;

        // Rounding
        uint32_t round = xbits & ((fp8_fp_half_ulp << 1) - 1);
        if ((round > fp8_fp_half_ulp) || ((round == fp8_fp_half_ulp) && (mantissa & 1))) {
            res += 1;
        }
    } else {
        // Denormal numbers
        uint8_t shift = 1 - exp;
        mantissa |= (1 << (fp8_significand_bits - 1));
        res = mantissa >> shift;

        // Rounding
        uint32_t round = (xbits | (1U << 22)) & ((fp8_fp_half_ulp << (shift + 1)) - 1);
        if ((round > (fp8_fp_half_ulp << shift)) || 
            ((round == (fp8_fp_half_ulp << shift)) && (res & 1))) {
            res += 1;
        }
    }

    // Combine result with sign
    return res | sign;
}

ushort fp8e4m3_to_fp16(uchar x) {
    ushort ur = (ushort(x) << 8);

    ushort sign = ur & 0x8000U;
    ushort exponent = static_cast<ushort>(((ur & 0x7800U) >> 1) + 0x2000U);
    ushort mantissa = static_cast<ushort>((ur & 0x0700U) >> 1);
    uchar absx = 0x7FU & static_cast<uchar>(x);

    if (absx == 0x7FU) {
        // NaN
        ur = 0x7FFFU; // fp16 canonical NaN, discard sign
    } else if (exponent == 0x2000U) {
        // zero or denormal
        if (mantissa != 0U) {
            // normalize
            mantissa = static_cast<ushort>(mantissa << 1);
            while ((mantissa & 0x0400U) == 0U) {
                mantissa = static_cast<ushort>(mantissa << 1);
                exponent = static_cast<ushort>(exponent - 0x0400U);
            }
            // discard implicit leading bit
            mantissa &= 0x03FFU;
        } else {
            // Zero
            exponent = 0U;
        }

        ur = (sign | exponent) | mantissa;
    } else {
        ur = (sign | exponent) | mantissa;
    }

    return ur;
}

kernel void cast_bf16_f8e4m3(
    constant size_t &dim,
    device const bfloat *input,
    device uint8_t *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) {
        return;
    }
    float x = static_cast<float>(input[tid]);
    output[tid] = f32_to_fp8e4m3(x);
}

kernel void cast_f8e4m3_bf16(
    constant size_t &dim,
    device const uint8_t *input,
    device bfloat *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) {
        return;
    }
    half x = as_type<half>(fp8e4m3_to_fp16(input[tid]));
    output[tid] = static_cast<bfloat>(x);
}

kernel void cast_f32_f8e4m3(
    constant size_t &dim,
    device const float *input,
    device uint8_t *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) {
        return;
    }
    float x = input[tid];
    output[tid] = f32_to_fp8e4m3(x);
}

kernel void cast_f8e4m3_f32(
    constant size_t &dim,
    device const uint8_t *input,
    device float *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) {
        return;
    }
    half x = as_type<half>(fp8e4m3_to_fp16(input[tid]));
    output[tid] = static_cast<float>(x);
}

// u32
CAST(cast_u32_f32, cast_u32_f32_strided, uint32_t, float)
CAST(cast_u32_u8, cast_u32_u8_strided, uint32_t, uint8_t)
CAST(cast_u32_f16, cast_u32_f16_strided, uint32_t, half)
CAST(cast_u32_i32, cast_u32_i32_strided, uint32_t, int32_t)
CAST(cast_u32_i16, cast_u32_i16_strided, uint32_t, int16_t)
#if __METAL_VERSION__ >= 220
CAST(cast_u32_i64, cast_u32_i64_strided, uint32_t, int64_t)
#endif
#if defined(__HAVE_BFLOAT__)
CAST(cast_u32_bf16, cast_u32_bf16_strided, uint32_t, bfloat)
#endif

// u8
CAST(cast_u8_u32, cast_u8_u32_strided, uint8_t, uint32_t)
CAST(cast_u8_f32, cast_u8_f32_strided, uint8_t, float)
CAST(cast_u8_f16, cast_u8_f16_strided, uint8_t, half)
CAST(cast_u8_i32, cast_u8_i32_strided, uint8_t, int64_t)
CAST(cast_u8_i16, cast_u8_i16_strided, uint8_t, int16_t)
#if __METAL_VERSION__ >= 220
CAST(cast_u8_i64, cast_u8_i64_strided, uint8_t, int64_t)
#endif
#if defined(__HAVE_BFLOAT__)
CAST(cast_u8_bf16, cast_u8_bf16_strided, uint8_t, bfloat)
#endif

// f16
CAST(cast_f16_f32, cast_f16_f32_strided, half, float)
CAST(cast_f16_u8, cast_f16_u8_strided, half, uint8_t)
CAST(cast_f16_u32, cast_f16_u32_strided, half, uint32_t)
CAST(cast_f16_i16, cast_f16_i16_strided, half, int16_t)
CAST(cast_f16_i32, cast_f16_i32_strided, half, int64_t)
CAST(cast_f16_i64, cast_f16_i64_strided, half, int64_t)
#if defined(__HAVE_BFLOAT__)
CAST_THROUGH(cast_f16_bf16, cast_f16_bf16_strided, half, bfloat, float)
#endif

// i64
CAST(cast_i64_f32, cast_i64_f32_strided, int64_t, float)
CAST(cast_i64_u8, cast_i64_u8_strided, int64_t, uint8_t)
CAST(cast_i64_u32, cast_i64_u32_strided, int64_t, uint32_t)
CAST(cast_i64_i32, cast_i64_i32_strided, int64_t, int32_t)
CAST(cast_i64_i16, cast_i64_i16_strided, int64_t, int16_t)
CAST(cast_i64_f16, cast_i64_f16_strided, int64_t, half)
#if defined(__HAVE_BFLOAT__)
CAST_THROUGH(cast_i64_bf16, cast_i64_bf16_strided, int64_t, bfloat, float)
#endif

// i32
CAST(cast_i32_f32, cast_i32_f32_strided, int32_t, float)
CAST(cast_i32_u8, cast_i32_u8_strided, int32_t, uint8_t)
CAST(cast_i32_u32, cast_i32_u32_strided, int32_t, uint32_t)
CAST(cast_i32_i64, cast_i32_i64_strided, int32_t, int64_t)
CAST(cast_i32_i16, cast_i32_i16_strided, int32_t, int16_t)
CAST(cast_i32_f16, cast_i32_f16_strided, int32_t, half)
#if defined(__HAVE_BFLOAT__)
CAST_THROUGH(cast_i32_bf16, cast_i32_bf16_strided, int64_t, bfloat, float)
#endif

// i16
CAST(cast_i16_f32, cast_i16_f32_strided, int16_t, float)
CAST(cast_i16_u8, cast_i16_u8_strided, int16_t, uint8_t)
CAST(cast_i16_u32, cast_i16_u32_strided, int16_t, uint32_t)
CAST(cast_i16_i32, cast_i16_i32_strided, int16_t, int32_t)
CAST(cast_i16_i64, cast_i16_i64_strided, int16_t, int64_t)
CAST(cast_i16_f16, cast_i16_f16_strided, int16_t, half)
#if defined(__HAVE_BFLOAT__)
CAST_THROUGH(cast_i16_bf16, cast_i16_bf16_strided, int16_t, bfloat, float)
#endif

// f32
CAST(cast_f32_f16, cast_f32_f16_strided, float, half)
CAST(cast_f32_u32, cast_f32_u32_strided, float, uint32_t)
CAST(cast_f32_u8, cast_f32_u8_strided, float, uint8_t)
CAST(cast_f32_i16, cast_f32_i16_strided, float, int16_t)
CAST(cast_f32_i32, cast_f32_i32_strided, float, int32_t)
CAST(cast_f32_i64, cast_f32_i64_strided, float, int64_t)
#if defined(__HAVE_BFLOAT__)
CAST(cast_f32_bf16, cast_f32_bf16_strided, float, bfloat)
#endif

// bf16
#if defined(__HAVE_BFLOAT__)
CAST(cast_bf16_u32, cast_bf16_u32_strided, bfloat, uint32_t)
CAST(cast_bf16_i16, cast_bf16_i16_strided, bfloat, int16_t)
CAST(cast_bf16_i32, cast_bf16_i32_strided, bfloat, int32_t)
CAST(cast_bf16_i64, cast_bf16_i64_strided, bfloat, int64_t)
CAST(cast_bf16_f32, cast_bf16_f32_strided, bfloat, float)
CAST_THROUGH(cast_bf16_u8, cast_bf16_u8_strided, bfloat, uint8_t, float)
CAST_THROUGH(cast_bf16_f16, cast_bf16_f16_strided, bfloat, half, float)
#endif