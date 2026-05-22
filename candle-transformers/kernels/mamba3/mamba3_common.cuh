#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef MAMBA3_CUDA_CHECK
#define MAMBA3_CUDA_CHECK(call) call
#endif

template<typename T>
__device__ __forceinline__ T mamba3_silu(T x) {
    return x / (T(1) + expf(-x));
}

__device__ __forceinline__ float mamba3_sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

__device__ __forceinline__ float mamba3_tanh(float x) {
    return tanhf(x);
}

__device__ __forceinline__ void mamba3_rope_pairwise(
    const float* x_in,
    float* x_out,
    const float* angles,
    int d_qk
) {
    int half = d_qk / 2;
    for (int i = 0; i < half; ++i) {
        float c = cosf(angles[i]);
        float s = sinf(angles[i]);
        float x0 = x_in[i];
        float x1 = x_in[i + half];
        x_out[i] = x0 * c - x1 * s;
        x_out[i + half] = x0 * s + x1 * c;
    }
}
