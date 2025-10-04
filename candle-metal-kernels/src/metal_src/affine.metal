#include <metal_stdlib>

template<uint D>
METAL_FUNC uint get_strided_idx(
    uint idx,
    constant const size_t *shape,
    constant const size_t *strides
);

template<>
METAL_FUNC uint get_strided_idx<1>(
    uint idx,
    constant const size_t *shape,
    constant const size_t *strides
) {
    return (idx % shape[0]) * strides[0];
}

template<>
METAL_FUNC uint get_strided_idx<2>(
    uint idx,
    constant const size_t *shape,
    constant const size_t *strides
) {
    return (idx % shape[1]) * strides[1]
    + ((idx / shape[1]) % shape[0]) * strides[0];
}

template<>
METAL_FUNC uint get_strided_idx<3>(
    uint idx,
    constant const size_t *shape,
    constant const size_t *strides
) {
    return (idx % shape[2]) * strides[2]
    + ((idx / shape[2]) % shape[1]) * strides[1]
    + ((idx / shape[2] / shape[1]) % shape[0]) * strides[0];
}

template<>
METAL_FUNC uint get_strided_idx<4>(
    uint idx,
    constant const size_t *shape,
    constant const size_t *strides
) {
    return (idx % shape[3]) * strides[3]
    + ((idx / shape[3]) % shape[2]) * strides[2]
    + ((idx / shape[3] / shape[2]) % shape[1]) * strides[1]
    + ((idx / shape[3] / shape[2] / shape[1]) % shape[0]) * strides[0];
}

template<>
METAL_FUNC uint get_strided_idx<5>(
    uint idx,
    constant const size_t *shape,
    constant const size_t *strides
) {
    return (idx % shape[4]) * strides[4]
    + ((idx / shape[4]) % shape[3]) * strides[3]
    + ((idx / shape[4] / shape[3]) % shape[2]) * strides[2]
    + ((idx / shape[4] / shape[3] / shape[2]) % shape[1]) * strides[1]
    + ((idx / shape[4] / shape[3] / shape[2] / shape[1]) % shape[0]) * strides[0];
}

template<>
METAL_FUNC uint get_strided_idx<6>(
    uint idx,
    constant const size_t *shape,
    constant const size_t *strides
) {
    return (idx % shape[5]) * strides[5]
    + ((idx / shape[5]) % shape[4]) * strides[4]
    + ((idx / shape[5] / shape[4]) % shape[3]) * strides[3]
    + ((idx / shape[5] / shape[4] / shape[3]) % shape[2]) * strides[2]
    + ((idx / shape[5] / shape[4] / shape[3] / shape[2]) % shape[1]) * strides[1]
    + ((idx / shape[5] / shape[4] / shape[3] / shape[2] / shape[1]) % shape[0]) * strides[0];
}

METAL_FUNC uint get_strided_index(
    uint idx,
    constant const size_t &num_dims,
    constant const size_t *dims,
    constant const size_t *strides
) {
    switch (num_dims) {
        case 1: return get_strided_idx<1>(idx, dims, strides);
        case 2: return get_strided_idx<2>(idx, dims, strides);
        case 3: return get_strided_idx<3>(idx, dims, strides);
        case 4: return get_strided_idx<4>(idx, dims, strides);
        case 5: return get_strided_idx<5>(idx, dims, strides);
        case 6: return get_strided_idx<6>(idx, dims, strides);
        default: {
            uint strided_i = 0;
            #pragma clang loop unroll(full)
            for (uint d = 0; d < num_dims; d++) {
                uint dim_idx = num_dims - 1 - d;
                strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
                idx /= dims[dim_idx];
            }
            return strided_i;
        }
    }
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
