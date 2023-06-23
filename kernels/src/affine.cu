#include "cuda_utils.cuh"
#include<stdint.h>

#define AFFINE_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *x, \
    TYPENAME *y, \
    const TYPENAME mul, \
    const TYPENAME add \
) {  \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;  \
    if (i >= numel) {  \
        return;  \
    }  \
    unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
    y[strided_i] = x[i] * mul + add; \
} \

AFFINE_OP(float, affine_f32)
AFFINE_OP(double, affine_f64)
AFFINE_OP(uint32_t, affine_u32)

