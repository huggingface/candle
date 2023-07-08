#include "cuda_utils.cuh"
#include<stdint.h>

#define CONV1D_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *info, \
    const uint32_t *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t h_size, \
    const size_t v_size \
) {  \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            memcpy(&out[i * h_size], &inp[ids[i] * h_size], h_size * sizeof(TYPENAME)); \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            memcpy(&out[i * h_size], &inp[ids[strided_i] * h_size], h_size * sizeof(TYPENAME)); \
        } \
    } \
} \

#if __CUDA_ARCH__ >= 800
CONV1D_OP(__nv_bfloat16, conv1d_bf16)
#endif

#if __CUDA_ARCH__ >= 530
CONV1D_OP(__half, conv1d_f16)
#endif

CONV1D_OP(float, conv1d_f32)
CONV1D_OP(double, conv1d_f64)
CONV1D_OP(uint8_t, conv1d_u8)
CONV1D_OP(uint32_t, conv1d_u32)

