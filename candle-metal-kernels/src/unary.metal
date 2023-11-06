#include <metal_stdlib>

struct Info{
    device size_t &num_dims;
    device size_t *dims;
    device size_t *strides;
};

METAL_FUNC uint get_strided_index(
    uint idx,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant size_t &offset
) {
    uint strided_i = offset;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}


using namespace metal;

#define UNARY(FN, TYPENAME, FN_NAME, FN_NAME_STRIDED) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint threadgroup_size [[threads_per_threadgroup]], \
    uint thread_index [[thread_index_in_threadgroup]] \
) { \
    const size_t length = (dim  + threadgroup_size - 1) / threadgroup_size; \
    const size_t start = thread_index * length; \
    const size_t stop = min(start + length, dim); \
    for (size_t i = start; i < stop; i++){ \
        output[i] = FN(input[i]); \
    } \
}\
kernel void FN_NAME_STRIDED( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    constant size_t &offset, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint threadgroup_size [[threads_per_threadgroup]], \
    uint thread_index [[thread_index_in_threadgroup]] \
) { \
    const size_t length = (dim  + threadgroup_size - 1) / threadgroup_size; \
    const size_t start = thread_index * length; \
    const size_t stop = min(start + length, dim); \
    for (size_t i = start; i < stop; i++){ \
        output[i] = FN(input[get_strided_index(i, num_dims, dims, strides, offset)]); \
    } \
}

UNARY(cos, float, cos_float, cos_float_strided);
UNARY(cos, half, cos_half, cos_half_strided);
UNARY(sin, float, sin_float, sin_float_strided);
UNARY(sin, half, sin_half, sin_half_strided);

#if __METAL_VERSION__ >= 310
UNARY(cos, bfloat, cos_bfloat, cos_bfloat_strided);
UNARY(sin, bfloat, sin_bfloat, sin_bfloat_strided);
#endif
