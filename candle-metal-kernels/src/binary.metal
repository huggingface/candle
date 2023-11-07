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
    uint threadgroup_size [[threads_per_threadgroup]], \
    uint thread_index [[thread_index_in_threadgroup]] \
) { \
    const size_t length = (dim  + threadgroup_size - 1) / threadgroup_size; \
    const size_t start = thread_index * length; \
    const size_t stop = min(start + length, dim); \
    for (size_t i = start; i < stop; i++){ \
        TYPENAME x = left[i]; \
        TYPENAME y = right[i]; \
        output[i] = OUT_TYPENAME(FN); \
    } \
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
    uint threadgroup_size [[threads_per_threadgroup]], \
    uint thread_index [[thread_index_in_threadgroup]] \
) { \
    const size_t length = (dim  + threadgroup_size - 1) / threadgroup_size; \
    const size_t start = thread_index * length; \
    const size_t stop = min(start + length, dim); \
    for (size_t i = start; i < stop; i++){ \
        TYPENAME x = left[get_strided_index(i, num_dims, dims, left_strides)]; \
        TYPENAME y = left[get_strided_index(i, num_dims, dims, right_strides)]; \
        output[i] = OUT_TYPENAME(FN); \
    } \
}

#define BINARY_OP(FN, NAME) \
BINARY(FN, float, float, NAME##_float, NAME##_float_strided); \
BINARY(FN, half, half, NAME##_half, NAME##_half_strided);

#define BFLOAT_BINARY_OP(FN, NAME) \
BINARY(NAME, bfloat, bfloat, NAME##_bfloat, NAME##_bfloat_strided);


BINARY_OP(x + y, add)
BINARY_OP(x - y, sub)
BINARY_OP(x * y, mul)
BINARY_OP(x / y, div)

#if __METAL_VERSION__ >= 310
BFLOAT_BINARY_OP(x + y, badd)
BFLOAT_BINARY_OP(x - y, bsub)
BFLOAT_BINARY_OP(x * y, bmul)
BFLOAT_BINARY_OP(x / y, bdiv)
#endif
