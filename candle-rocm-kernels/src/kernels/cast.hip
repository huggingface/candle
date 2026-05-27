#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

struct __rocm_bf16 {
    uint16_t __x;

    __device__ __rocm_bf16() : __x(0) {}

    __device__ __rocm_bf16(float f) {
        unsigned int u;
        memcpy(&u, &f, sizeof(u));
        __x = (uint16_t)(u >> 16);
    }

    __device__ __rocm_bf16(double d) {
        float f = (float)d;
        unsigned int u;
        memcpy(&u, &f, sizeof(u));
        __x = (uint16_t)(u >> 16);
    }

    __device__ operator float() const {
        float f = 0.0f;
        unsigned int u = (unsigned int)__x << 16;
        memcpy(&f, &u, sizeof(f));
        return f;
    }
};

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
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = (DST_TYPENAME)inp[i]; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int strided_i = get_strided_index(i, num_dims, dims, strides); \
            out[i] = (DST_TYPENAME)inp[strided_i]; \
        } \
    } \
}

#define CAST_THROUGH_OP(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const SRC_TYPENAME *inp, \
    DST_TYPENAME *out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = (DST_TYPENAME)(float)inp[i]; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int strided_i = get_strided_index(i, num_dims, dims, strides); \
            out[i] = (DST_TYPENAME)(float)inp[strided_i]; \
        } \
    } \
}

CAST_OP(uint8_t, uint8_t, cast_u8_u8)
CAST_OP(uint8_t, uint32_t, cast_u8_u32)
CAST_OP(uint8_t, int64_t, cast_u8_i64)
CAST_OP(uint8_t, float, cast_u8_f32)
CAST_OP(uint8_t, double, cast_u8_f64)

CAST_OP(uint32_t, uint8_t, cast_u32_u8)
CAST_OP(uint32_t, uint32_t, cast_u32_u32)
CAST_OP(uint32_t, int64_t, cast_u32_i64)
CAST_OP(uint32_t, float, cast_u32_f32)
CAST_OP(uint32_t, double, cast_u32_f64)

CAST_OP(int64_t, uint8_t, cast_i64_u8)
CAST_OP(int64_t, uint32_t, cast_i64_u32)
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

CAST_THROUGH_OP(__rocm_bf16, __rocm_bf16, cast_bf16_bf16)
CAST_THROUGH_OP(__rocm_bf16, uint8_t, cast_bf16_u8)
CAST_THROUGH_OP(__rocm_bf16, uint32_t, cast_bf16_u32)
CAST_THROUGH_OP(__rocm_bf16, int64_t, cast_bf16_i64)
CAST_THROUGH_OP(__rocm_bf16, float, cast_bf16_f32)
CAST_THROUGH_OP(__rocm_bf16, double, cast_bf16_f64)
CAST_THROUGH_OP(__rocm_bf16, half, cast_bf16_f16)

CAST_THROUGH_OP(uint8_t, __rocm_bf16, cast_u8_bf16)
CAST_THROUGH_OP(uint32_t, __rocm_bf16, cast_u32_bf16)
CAST_THROUGH_OP(int64_t, __rocm_bf16, cast_i64_bf16)
CAST_THROUGH_OP(float, __rocm_bf16, cast_f32_bf16)
CAST_THROUGH_OP(double, __rocm_bf16, cast_f64_bf16)
CAST_THROUGH_OP(half, __rocm_bf16, cast_f16_bf16)

CAST_THROUGH_OP(half, half, cast_f16_f16)
CAST_THROUGH_OP(half, uint8_t, cast_f16_u8)
CAST_THROUGH_OP(half, uint32_t, cast_f16_u32)
CAST_THROUGH_OP(half, int64_t, cast_f16_i64)
CAST_THROUGH_OP(half, float, cast_f16_f32)
CAST_THROUGH_OP(half, double, cast_f16_f64)

CAST_THROUGH_OP(uint8_t, half, cast_u8_f16)
CAST_THROUGH_OP(uint32_t, half, cast_u32_f16)
CAST_THROUGH_OP(int64_t, half, cast_i64_f16)
CAST_THROUGH_OP(float, half, cast_f32_f16)
CAST_THROUGH_OP(double, half, cast_f64_f16)
