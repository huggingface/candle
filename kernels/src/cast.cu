#include "cuda_utils.cuh"

#define CAST_OP(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const SRC_TYPENAME *inp, \
    DST_TYPENAME *out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = inp[i]; \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            out[i] = inp[strided_i]; \
        } \
    } \
} \

#if __CUDA_ARCH__ >= 530
CAST_OP(__half, __half, cast_f16_f16)
CAST_OP(__half, float, cast_f16_f32)
CAST_OP(float, __half, cast_f32_f16)
#endif

CAST_OP(float, float, cast_f32_f32)
CAST_OP(float, double, cast_f32_f64)
CAST_OP(double, float, cast_f64_f32)
