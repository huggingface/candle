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

#define AFFINE(FN_NAME, TYPENAME) \
kernel void FN_NAME( \
    constant size_t &dim, \
    constant float &mul, \
    constant float &add, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]], \
    uint thread_position_in_threadgroup [[ thread_position_in_threadgroup ]], \
    uint threads_per_threadgroup        [[ threads_per_threadgroup ]] \
) { \
    uint thread_position_in_grid = (threadgroup_position_in_grid * threads_per_threadgroup) + \
        thread_position_in_threadgroup; \
    if (thread_position_in_grid >= dim) { \
        return; \
    } \
    const TYPENAME m = TYPENAME(mul); \
    const TYPENAME a = TYPENAME(add); \
    output[thread_position_in_grid] = input[thread_position_in_grid] * m + a; \
} \

AFFINE(affine_float, float)
AFFINE(affine_half, half)


#if __METAL_VERSION__ >= 310
AFFINE(affine_bfloat, bfloat);
#endif
