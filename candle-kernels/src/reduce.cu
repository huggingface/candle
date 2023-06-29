// TODO: Use a proper distributed reduction rather than atomicAdd.
// https://people.maths.ox.ac.uk/gilesm/cuda/prac4/reduction.pdf
#include "cuda_utils.cuh"
#include<stdint.h>

#define SUM_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t num_sum_dims, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out \
) {  \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    const size_t *sum_dims_l = info + 2*num_dims; \
    const size_t *sum_dims_s = info + 2*num_dims + num_sum_dims; \
    if (is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            size_t dst_index = i; \
            for (unsigned int nd = 0; nd < num_sum_dims; ++nd) { \
              size_t stride = sum_dims_s[nd]; \
              size_t pre = dst_index / stride; \
              size_t post = dst_index % stride; \
              dst_index = (pre / sum_dims_l[nd]) * stride + post; \
            } \
            atomicAdd(out + dst_index, inp[i]); \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            size_t dst_index = i; \
            for (unsigned int nd = 0; nd < num_sum_dims; ++nd) { \
              size_t stride = sum_dims_s[nd]; \
              size_t pre = dst_index / stride; \
              size_t post = dst_index % stride; \
              dst_index = (pre / sum_dims_l[nd]) * stride + post; \
            } \
            atomicAdd(out + dst_index, inp[strided_i]); \
        } \
    } \
} \

#if __CUDA_ARCH__ >= 800
SUM_OP(__nv_bfloat16, sum_bf16)
#endif

#if __CUDA_ARCH__ >= 530
SUM_OP(__half, sum_f16)
#endif

SUM_OP(float, sum_f32)
SUM_OP(double, sum_f64)
SUM_OP(uint32_t, sum_u32)
