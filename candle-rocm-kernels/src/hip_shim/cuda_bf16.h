// Shadows CUDA's <cuda_bf16.h> for the shared candle-kernels sources.
#pragma once

#include <hip/hip_bf16.h>

// The only spelling difference between the two bfloat16 APIs. Everything the
// kernels use (__float2bfloat16, __bfloat162float, __hmax, __hmin, arithmetic
// operators) already matches.
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;

// CUDA spells the NaN-propagating min/max with a `_nan` suffix. HIP's plain
// __hmax/__hmin on bfloat16 already propagate NaN, but the suffixed names do
// not exist, and compatibility.cuh only declares the __half variants (and only
// for __CUDA_ARCH__ < 800, which we compile past).
__device__ __forceinline__ __nv_bfloat16 __hmax_nan(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmax(a, b));
}

__device__ __forceinline__ __nv_bfloat16 __hmin_nan(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hisnan(a) ? a : (__hisnan(b) ? b : __hmin(a, b));
}
