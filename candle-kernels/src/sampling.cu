// CUDA implementation of repeat-penalty sampling op.
// Mirrors the Metal and CPU implementations in candle-core/src/sampling.rs:
//   logit >= 0 -> logit / penalty
//   logit <  0 -> logit * penalty
// applied to every vocab position whose index appears in `context`.

#include "cuda_fp16.h"
#include <stdint.h>

template <typename T>
__device__ void repeat_penalty_impl(
    const T *input,
    T *output,
    const uint32_t *context,
    const uint32_t vocab_size,
    const uint32_t context_size,
    const float penalty
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab_size) {
        return;
    }

    float logit = static_cast<float>(input[i]);

    // Linear scan: vocab positions are independent so this avoids
    // atomics / scatter conflicts. `context` is expected to be deduped.
    for (uint32_t j = 0; j < context_size; ++j) {
        if (context[j] == i) {
            logit = (logit >= 0.0f) ? (logit / penalty) : (logit * penalty);
            break;
        }
    }

    output[i] = static_cast<T>(logit);
}

extern "C" __global__ void repeat_penalty_f32(
    const float *input,
    float *output,
    const uint32_t *context,
    const uint32_t vocab_size,
    const uint32_t context_size,
    const float penalty
) {
    repeat_penalty_impl<float>(input, output, context, vocab_size, context_size, penalty);
}

extern "C" __global__ void repeat_penalty_f64(
    const double *input,
    double *output,
    const uint32_t *context,
    const uint32_t vocab_size,
    const uint32_t context_size,
    const float penalty
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab_size) {
        return;
    }
    double logit = input[i];
    const double p = static_cast<double>(penalty);
    for (uint32_t j = 0; j < context_size; ++j) {
        if (context[j] == i) {
            logit = (logit >= 0.0) ? (logit / p) : (logit * p);
            break;
        }
    }
    output[i] = logit;
}

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void repeat_penalty_f16(
    const __half *input,
    __half *output,
    const uint32_t *context,
    const uint32_t vocab_size,
    const uint32_t context_size,
    const float penalty
) {
    repeat_penalty_impl<__half>(input, output, context, vocab_size, context_size, penalty);
}
#endif

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>

extern "C" __global__ void repeat_penalty_bf16(
    const __nv_bfloat16 *input,
    __nv_bfloat16 *output,
    const uint32_t *context,
    const uint32_t vocab_size,
    const uint32_t context_size,
    const float penalty
) {
    repeat_penalty_impl<__nv_bfloat16>(input, output, context, vocab_size, context_size, penalty);
}
#endif
