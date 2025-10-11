#include <metal_stdlib>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

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

#define BINARY(FN, TYPENAME, OUT_TYPENAME, FN_NAME, FN_NAME_STRIDED) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const TYPENAME *left,  \
    device const TYPENAME *right,  \
    device OUT_TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    TYPENAME x = left[tid]; \
    TYPENAME y = right[tid]; \
    output[tid] = OUT_TYPENAME(FN); \
}\
kernel void FN_NAME_STRIDED( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *left_strides, \
    constant size_t *right_strides, \
    device const TYPENAME *left,  \
    device const TYPENAME *right,  \
    device OUT_TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    TYPENAME x = left[get_strided_index(tid, num_dims, dims, left_strides)]; \
    TYPENAME y = right[get_strided_index(tid, num_dims, dims, right_strides)]; \
    output[tid] = OUT_TYPENAME(FN); \
}

#define BINARY_OP(FN, NAME) \
BINARY(FN, float, float, NAME##_f32, NAME##_f32_strided); \
BINARY(FN, half, half, NAME##_f16, NAME##_f16_strided); \
BINARY(FN, uint32_t, uint32_t, NAME##_u32, NAME##_u32_strided); \
BINARY(FN, uint8_t, uint8_t, NAME##_u8, NAME##_u8_strided);

#define BINARY_OP_OUT(NAME, FN) \
BINARY(FN, float, uint8_t, NAME##_f32, NAME##_f32_strided); \
BINARY(FN, half, uint8_t, NAME##_f16, NAME##_f16_strided); \
BINARY(FN, uint32_t, uint8_t, NAME##_u32, NAME##_u32_strided); \
BINARY(FN, uint8_t, uint8_t, NAME##_u8, NAME##_u8_strided);

#define INT64_BINARY_OP(NAME, FN) \
BINARY(FN, int64_t, int64_t, NAME##_i64, NAME##_i64_strided);

#define INT64_BINARY_OP_OUT(NAME, FN) \
BINARY(FN, int64_t, uint8_t, NAME##_i64, NAME##_i64_strided);

#define BFLOAT_BINARY_OP(FN, NAME) \
BINARY(FN, bfloat, bfloat, NAME##_bf16, NAME##_bf16_strided);

#define BFLOAT_BINARY_OP_OUT(NAME, FN) \
BINARY(FN, bfloat, uint8_t, NAME##_bf16, NAME##_bf16_strided);

BINARY_OP(x + y, add)
BINARY_OP(x - y, sub)
BINARY_OP(x * y, mul)
BINARY_OP(x / y, div)
BINARY_OP(MIN(x, y), min)
BINARY_OP(MAX(x, y), max)

BINARY_OP_OUT(eq, x == y)
BINARY_OP_OUT(ne, x != y)
BINARY_OP_OUT(le, x <= y)
BINARY_OP_OUT(lt, x < y)
BINARY_OP_OUT(ge, x >= y)
BINARY_OP_OUT(gt, x > y)

#if __METAL_VERSION__ >= 220
INT64_BINARY_OP(add, x + y)
INT64_BINARY_OP(sub, x - y)
INT64_BINARY_OP(mul, x * y)
INT64_BINARY_OP(div, x / y)
INT64_BINARY_OP(min, MIN(x, y))
INT64_BINARY_OP(max, MAX(x, y))

INT64_BINARY_OP_OUT(eq, x == y)
INT64_BINARY_OP_OUT(ne, x != y)
INT64_BINARY_OP_OUT(le, x <= y)
INT64_BINARY_OP_OUT(lt, x < y)
INT64_BINARY_OP_OUT(ge, x >= y)
INT64_BINARY_OP_OUT(gt, x > y)
#endif

#if defined(__HAVE_BFLOAT__)
BFLOAT_BINARY_OP(x + y, add)
BFLOAT_BINARY_OP(x - y, sub)
BFLOAT_BINARY_OP(x * y, mul)
BFLOAT_BINARY_OP(x / y, div)
BFLOAT_BINARY_OP(MIN(x, y), min)
BFLOAT_BINARY_OP(MAX(x, y), max)

BFLOAT_BINARY_OP_OUT(eq, x == y)
BFLOAT_BINARY_OP_OUT(ne, x != y)
BFLOAT_BINARY_OP_OUT(le, x <= y)
BFLOAT_BINARY_OP_OUT(lt, x < y)
BFLOAT_BINARY_OP_OUT(ge, x >= y)
BFLOAT_BINARY_OP_OUT(gt, x > y)
#endif
