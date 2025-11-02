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

// u32
CAST(cast_u32_f32, cast_u32_f32_strided, uint32_t, float)
CAST(cast_u32_u8, cast_u32_u8_strided, uint32_t, uint8_t)
CAST(cast_u32_f16, cast_u32_f16_strided, uint32_t, half)
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
CAST(cast_f16_i64, cast_f16_i64_strided, half, int64_t)
#if defined(__HAVE_BFLOAT__)
CAST_THROUGH(cast_f16_bf16, cast_f16_bf16_strided, half, bfloat, float)
#endif

// i64
CAST(cast_i64_f32, cast_i64_f32_strided, int64_t, float)
CAST(cast_i64_u8, cast_i64_u8_strided, int64_t, uint8_t)
CAST(cast_i64_u32, cast_i64_u32_strided, int64_t, uint32_t)
CAST(cast_i64_f16, cast_i64_f16_strided, int64_t, half)
#if defined(__HAVE_BFLOAT__)
CAST_THROUGH(cast_i64_bf16, cast_i64_bf16_strided, int64_t, bfloat, float)
#endif

// f32
CAST(cast_f32_f16, cast_f32_f16_strided, float, half)
CAST(cast_f32_u32, cast_f32_u32_strided, float, uint32_t)
CAST(cast_f32_u8, cast_f32_u8_strided, float, uint8_t)
CAST(cast_f32_i64, cast_f32_i64_strided, float, int64_t)
#if defined(__HAVE_BFLOAT__)
CAST(cast_f32_bf16, cast_f32_bf16_strided, float, bfloat)
#endif

// bf16
#if defined(__HAVE_BFLOAT__)
CAST(cast_bf16_u32, cast_bf16_u32_strided, bfloat, uint32_t)
CAST(cast_bf16_i64, cast_bf16_i64_strided, bfloat, int64_t)
CAST(cast_bf16_f32, cast_bf16_f32_strided, bfloat, float)
CAST_THROUGH(cast_bf16_u8, cast_bf16_u8_strided, bfloat, uint8_t, float)
CAST_THROUGH(cast_bf16_f16, cast_bf16_f16_strided, bfloat, half, float)
#endif