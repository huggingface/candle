#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#endif

#include <stddef.h>
#include <stdint.h>
#include <math.h>

__device__ bool is_contiguous(
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    size_t acc = 1;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        if (dims[dim_idx] > 1 && acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    }
    return true;
}

__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    unsigned int tmp_i = idx;
    for (int d = num_dims - 1; d >= 0; d--) {
        unsigned int i_dim = tmp_i % dims[d];
        strided_i += i_dim * strides[d];
        tmp_i /= dims[d];
    }
    return strided_i;
}

#define AFFINE_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const TYPENAME mul, \
    const TYPENAME add \
) { \
    if (info == nullptr || is_contiguous(num_dims, info, info + num_dims)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = inp[i] * mul + add; \
        } \
    } else { \
        const size_t *strides = info + num_dims; \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int src_i = get_strided_index(i, num_dims, info, strides); \
            out[i] = inp[src_i] * mul + add; \
        } \
    } \
}

AFFINE_OP(float, affine_f32)
AFFINE_OP(double, affine_f64)
AFFINE_OP(uint8_t, affine_u8)
AFFINE_OP(uint32_t, affine_u32)
AFFINE_OP(int64_t, affine_i64)
