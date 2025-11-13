// Translated from affine.cu by TEAM-488 (Phase 2)
// HIP kernel for affine transformations: out = x * mul + add

#include "hip_utils.h"
#include<stdint.h>

#define AFFINE_OP(TYPENAME, FN_NAME, AFFINE) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const TYPENAME mul, \
    const TYPENAME add \
) {  \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = inp ? inp[i] : out[i]; \
            out[i] = AFFINE; \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            TYPENAME x = inp ? inp[strided_i] : out[i]; \
            out[i] = AFFINE; \
        } \
    } \
} \

// BF16 support (gfx90a and later)
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
AFFINE_OP(__hip_bfloat16, affine_bf16, x * mul + add)
#endif

// FP16 support (all modern AMD GPUs)
AFFINE_OP(_Float16, affine_f16, x * mul + add)

// Standard types (all AMD GPUs)
AFFINE_OP(float, affine_f32, x * mul + add)
AFFINE_OP(double, affine_f64, x * mul + add)
AFFINE_OP(uint8_t, affine_u8, x * mul + add)
AFFINE_OP(uint32_t, affine_u32, x * mul + add)
AFFINE_OP(int16_t, affine_i16, x * mul + add)
AFFINE_OP(int32_t, affine_i32, x * mul + add)
AFFINE_OP(int64_t, affine_i64, x * mul + add)

// Note: F8E4M3 support depends on CDNA3 (gfx940+)
// TODO: Add F8E4M3 support when ROCm provides the intrinsics
