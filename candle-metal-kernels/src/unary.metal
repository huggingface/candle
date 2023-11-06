#include <metal_stdlib>
#
METAL_FUNC bool is_contiguous(
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    size_t acc = 1;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        if (acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    }
    return true;
}

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

#define UNARY(FN, TYPENAME, FN_NAME, FN_NAME_STRIDED) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint threadgroup_size [[threads_per_threadgroup]], \
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]], \
    uint thread_index [[thread_index_in_threadgroup]] \
) { \
    const uint i = thread_index + (threadgroup_position_in_grid * threadgroup_size); \
    if (i > dim){ \
        return; \
    } \
    output[i] = FN(input[i]); \
}\
kernel void FN_NAME_STRIDED( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *info, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint threadgroup_size [[threads_per_threadgroup]], \
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]], \
    uint thread_index [[thread_index_in_threadgroup]] \
) { \
    constant size_t *dims = info; \
    constant size_t *strides = info + num_dims; \
    const uint start = thread_index + (threadgroup_position_in_grid * threadgroup_size); \
    const uint stop = min(thread_index + (threadgroup_position_in_grid * threadgroup_size), (uint) dim); \
    for (size_t i = start; i < stop; i++) { \
        output[i] = FN(input[get_strided_index(i, num_dims, dims, strides)]); \
    } \
}

UNARY(cos, float, cos_float, cos_float_strided);
UNARY(cos, half, cos_half, cos_half_strided);

#if __METAL_VERSION__ >= 310
UNARY(cos, bfloat, cos_bfloat, cos_bfloat_strided);
#endif
