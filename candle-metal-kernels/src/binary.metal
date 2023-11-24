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

#define BINARY(FN, TYPENAME, OUT_TYPENAME, FN_NAME, FN_NAME_STRIDED) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const TYPENAME *left,  \
    device const TYPENAME *right,  \
    device TYPENAME *output, \
    uint thread_position_in_grid [[ thread_position_in_grid ]] \
) { \
    if (thread_position_in_grid >= dim) { \
        return; \
    } \
    TYPENAME x = left[thread_position_in_grid]; \
    TYPENAME y = right[thread_position_in_grid]; \
    output[thread_position_in_grid] = OUT_TYPENAME(FN); \
}\
kernel void FN_NAME_STRIDED( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *left_strides, \
    constant size_t *right_strides, \
    device const TYPENAME *left,  \
    device const TYPENAME *right,  \
    device TYPENAME *output, \
    uint thread_position_in_grid [[ thread_position_in_grid ]] \
) { \
    if (thread_position_in_grid >= dim) { \
        return; \
    } \
    TYPENAME x = left[get_strided_index(thread_position_in_grid, num_dims, dims, left_strides)]; \
    TYPENAME y = right[get_strided_index(thread_position_in_grid, num_dims, dims, right_strides)]; \
    output[thread_position_in_grid] = OUT_TYPENAME(FN); \
}

#define BINARY_OP(FN, NAME) \
BINARY(FN, float, float, NAME##_float, NAME##_float_strided); \
BINARY(FN, half, half, NAME##_half, NAME##_half_strided);

#define BFLOAT_BINARY_OP(FN, NAME) \
BINARY(FN, bfloat, bfloat, NAME##_bfloat, NAME##_bfloat_strided);


BINARY_OP(x + y, add)
BINARY_OP(x - y, sub)
BINARY_OP(x * y, mul)
BINARY_OP(x / y, div)

#if __METAL_VERSION__ >= 310
BFLOAT_BINARY_OP(x + y, add)
BFLOAT_BINARY_OP(x - y, sub)
BFLOAT_BINARY_OP(x * y, mul)
BFLOAT_BINARY_OP(x / y, div)
#endif
