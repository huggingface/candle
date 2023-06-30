#include "cuda_utils.cuh"
#include<stdint.h>

#define WHERE_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *info, \
    const uint32_t *ids, \
    const TYPENAME *t, \
    const TYPENAME *f, \
    TYPENAME *out \
) {  \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    const size_t *strides_t = info + 2*num_dims; \
    const size_t *strides_f = info + 3*num_dims; \
    if (is_contiguous(num_dims, dims, strides) \
        && is_contiguous(num_dims, dims, strides_f) \
        && is_contiguous(num_dims, dims, strides_t)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = ids[i] ? t[i] : f[i]; \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            unsigned strided_i_t = get_strided_index(i, num_dims, dims, strides_t); \
            unsigned strided_i_f = get_strided_index(i, num_dims, dims, strides_f); \
            out[i] = ids[strided_i] ? t[strided_i_t] : f[strided_i_f]; \
        } \
    } \
} \

#if __CUDA_ARCH__ >= 800
WHERE_OP(__nv_bfloat16, where_bf16)
#endif

#if __CUDA_ARCH__ >= 530
WHERE_OP(__half, where_f16)
#endif

WHERE_OP(float, where_f32)
WHERE_OP(double, where_f64)
WHERE_OP(uint8_t, where_u8)
WHERE_OP(uint32_t, where_u32)
