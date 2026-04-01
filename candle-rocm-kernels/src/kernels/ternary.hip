#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#else
#include <hip/hip_runtime.h>
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

#define WHERE_OP(TYPENAME, ID_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const ID_TYPENAME *ids, \
    const TYPENAME *t, \
    const TYPENAME *f, \
    TYPENAME *out \
) { \
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
}

WHERE_OP(float, uint8_t, where_u8_f32)
WHERE_OP(double, uint8_t, where_u8_f64)
WHERE_OP(uint8_t, uint8_t, where_u8_u8)
WHERE_OP(uint32_t, uint8_t, where_u8_u32)
WHERE_OP(int64_t, uint8_t, where_u8_i64)

WHERE_OP(float, uint32_t, where_u32_f32)
WHERE_OP(double, uint32_t, where_u32_f64)
WHERE_OP(uint8_t, uint32_t, where_u32_u8)
WHERE_OP(uint32_t, uint32_t, where_u32_u32)
WHERE_OP(int64_t, uint32_t, where_u32_i64)

WHERE_OP(float, int64_t, where_i64_f32)
WHERE_OP(double, int64_t, where_i64_f64)
WHERE_OP(uint8_t, int64_t, where_i64_u8)
WHERE_OP(uint32_t, int64_t, where_i64_u32)
WHERE_OP(int64_t, int64_t, where_i64_i64)

WHERE_OP(float, uint16_t, where_u16_f32)
WHERE_OP(double, uint16_t, where_u16_f64)
WHERE_OP(uint8_t, uint16_t, where_u16_u8)
WHERE_OP(uint32_t, uint16_t, where_u16_u32)
WHERE_OP(int64_t, uint16_t, where_u16_i64)
