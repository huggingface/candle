#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#endif

#include <stddef.h>
#include <stdint.h>

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

#define FILL_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    TYPENAME *buf, \
    TYPENAME value, \
    const size_t numel \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        buf[i] = value; \
    } \
}

#define CONST_SET_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME inp, \
    TYPENAME *out \
) { \
    if (info == nullptr || is_contiguous(num_dims, info, info + num_dims)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = inp; \
        } \
    } else { \
        const size_t *strides = info + num_dims; \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int dst_i = get_strided_index(i, num_dims, info, strides); \
            out[dst_i] = inp; \
        } \
    } \
}

FILL_OP(float, fill_f32)
FILL_OP(double, fill_f64)
FILL_OP(uint8_t, fill_u8)
FILL_OP(uint32_t, fill_u32)
FILL_OP(int64_t, fill_i64)

CONST_SET_OP(float, const_set_f32)
CONST_SET_OP(double, const_set_f64)
CONST_SET_OP(uint8_t, const_set_u8)
CONST_SET_OP(uint32_t, const_set_u32)
CONST_SET_OP(int64_t, const_set_i64)
