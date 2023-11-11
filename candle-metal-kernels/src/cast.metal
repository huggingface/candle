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
    uint thread_position_in_grid [[ thread_position_in_grid ]] \
) { \
    if (thread_position_in_grid >= dim) { \
        return; \
    } \
    output[thread_position_in_grid] = RIGHT_TYPENAME(input[thread_position_in_grid]); \
} \
kernel void FN_NAME_STRIDED( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    device const LEFT_TYPENAME *input,  \
    device RIGHT_TYPENAME *output, \
    uint i [[ thread_position_in_grid ]] \
) { \
    if (i >= dim) { \
        return; \
    } \
    output[i] = RIGHT_TYPENAME(input[get_strided_index(i, num_dims, dims, strides)]); \
} \

CAST(cast_u32_f32, cast_u32_f32_strided, int32_t, float)
CAST(cast_f16_f32, cast_f16_f32_strided, half, float)
CAST(cast_f32_f16, cast_f32_f16_strided, float, half)

#if __METAL_VERSION__ >= 310
#endif
