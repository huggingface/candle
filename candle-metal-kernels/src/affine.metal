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

#define AFFINE(FN_NAME, T) \
kernel void FN_NAME( \
    constant size_t &dim, \
    constant float &mul, \
    constant float &add, \
    device const T *input,  \
    device T *output, \
    uint id [[ thread_position_in_grid ]] \
) { \
    if (id >= dim) { \
        return; \
    } \
    output[id] = T(fma(float(input[id]), mul, add)); \
} \
kernel void FN_NAME##_strided( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    constant float &mul, \
    constant float &add, \
    device const T *input,  \
    device T *output, \
    uint id [[ thread_position_in_grid ]] \
) { \
    if (id >= dim) { \
        return; \
    } \
    output[id] = T(fma(float(input[get_strided_index(id, num_dims, dims, strides)]), mul, add)); \
}

#define POWF(FN_NAME, TYPENAME) \
kernel void FN_NAME( \
    constant size_t &dim, \
    constant float &mul, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint id [[ thread_position_in_grid ]] \
) { \
    if (id >= dim) { \
        return; \
    } \
    output[id] = TYPENAME(pow(input[id], TYPENAME(mul))); \
} \
kernel void FN_NAME##_strided( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    constant float &mul, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint id [[ thread_position_in_grid ]] \
) { \
    if (id >= dim) { \
        return; \
    } \
    output[id] = TYPENAME(pow(input[get_strided_index(id, num_dims, dims, strides)], TYPENAME(mul))); \
}

#define ELU(FN_NAME, TYPENAME) \
kernel void FN_NAME( \
    constant size_t &dim, \
    constant float &mul, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint id [[ thread_position_in_grid ]] \
) { \
    if (id >= dim) { \
        return; \
    } \
    const TYPENAME x = input[id]; \
    output[id] = TYPENAME((x > 0)?x: mul * (exp(x) - 1)); \
} \
kernel void FN_NAME##_strided( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    constant float &mul, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint id [[ thread_position_in_grid ]] \
) { \
    if (id >= dim) { \
        return; \
    } \
    const TYPENAME x = input[get_strided_index(id, num_dims, dims, strides)]; \
    output[id] = TYPENAME((x > 0)?x: mul * (exp(x) - 1)); \
} \


AFFINE(affine_u8, uint8_t)
AFFINE(affine_u32, uint32_t)
AFFINE(affine_f32, float)
AFFINE(affine_f16, half)
POWF(powf_f32, float)
POWF(powf_f16, half)
ELU(elu_f32, float)
ELU(elu_f16, half)


#if defined(__HAVE_BFLOAT__)
AFFINE(affine_bf16, bfloat);
POWF(powf_bf16, bfloat);
ELU(elu_bf16, bfloat);
#endif
