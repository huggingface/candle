#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#else
#include <hip/hip_runtime.h>
#endif

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <hip/hip_fp16.h>

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

#define CAST_OP(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const SRC_TYPENAME *inp, \
    DST_TYPENAME *out \
) { \
    if (info == nullptr || is_contiguous(num_dims, info, info + num_dims)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = (DST_TYPENAME)(inp[i]); \
        } \
    } else { \
        const size_t *strides = info + num_dims; \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int src_i = get_strided_index(i, num_dims, info, strides); \
            out[i] = (DST_TYPENAME)(inp[src_i]); \
        } \
    } \
}

#define CAST_THROUGH(FLOAT_TYPENAME, SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const SRC_TYPENAME *inp, \
    DST_TYPENAME *out \
) { \
    if (info == nullptr || is_contiguous(num_dims, info, info + num_dims)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = (DST_TYPENAME)((FLOAT_TYPENAME)(inp[i])); \
        } \
    } else { \
        const size_t *strides = info + num_dims; \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int src_i = get_strided_index(i, num_dims, info, strides); \
            out[i] = (DST_TYPENAME)((FLOAT_TYPENAME)(inp[src_i])); \
        } \
    } \
}

CAST_OP(uint32_t, uint32_t, cast_u32_u32)
CAST_OP(uint32_t, uint8_t, cast_u32_u8)
CAST_OP(uint32_t, int64_t, cast_u32_i64)
CAST_OP(uint32_t, float, cast_u32_f32)
CAST_OP(uint32_t, double, cast_u32_f64)

CAST_OP(uint8_t, uint32_t, cast_u8_u32)
CAST_OP(uint8_t, uint8_t, cast_u8_u8)
CAST_OP(uint8_t, int64_t, cast_u8_i64)
CAST_OP(uint8_t, float, cast_u8_f32)
CAST_OP(uint8_t, double, cast_u8_f64)

CAST_OP(int64_t, uint32_t, cast_i64_u32)
CAST_OP(int64_t, uint8_t, cast_i64_u8)
CAST_OP(int64_t, int64_t, cast_i64_i64)
CAST_OP(int64_t, float, cast_i64_f32)
CAST_OP(int64_t, double, cast_i64_f64)

CAST_OP(float, uint8_t, cast_f32_u8)
CAST_OP(float, uint32_t, cast_f32_u32)
CAST_OP(float, int64_t, cast_f32_i64)
CAST_OP(float, float, cast_f32_f32)
CAST_OP(float, double, cast_f32_f64)

CAST_OP(double, uint8_t, cast_f64_u8)
CAST_OP(double, uint32_t, cast_f64_u32)
CAST_OP(double, int64_t, cast_f64_i64)
CAST_OP(double, float, cast_f64_f32)
CAST_OP(double, double, cast_f64_f64)

CAST_OP(float, __half, cast_f32_f16)
CAST_OP(double, __half, cast_f64_f16)
CAST_OP(__half, float, cast_f16_f32)
CAST_OP(__half, double, cast_f16_f64)
CAST_OP(__half, __half, cast_f16_f16)
CAST_OP(uint8_t, __half, cast_u8_f16)
CAST_OP(uint32_t, __half, cast_u32_f16)
CAST_THROUGH(float, __half, uint8_t, cast_f16_u8)

CAST_OP(float, __hip_bfloat16, cast_f32_bf16)
CAST_OP(double, __hip_bfloat16, cast_f64_bf16)
CAST_OP(__hip_bfloat16, float, cast_bf16_f32)
CAST_OP(__hip_bfloat16, double, cast_bf16_f64)
CAST_OP(__hip_bfloat16, __hip_bfloat16, cast_bf16_bf16)
CAST_OP(uint8_t, __hip_bfloat16, cast_u8_bf16)
CAST_OP(uint32_t, __hip_bfloat16, cast_u32_bf16)
CAST_THROUGH(float, __hip_bfloat16, uint8_t, cast_bf16_u8)
CAST_THROUGH(float, __hip_bfloat16, __half, cast_bf16_f16)
CAST_THROUGH(float, __half, __hip_bfloat16, cast_f16_bf16)
