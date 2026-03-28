#undef __CUDA_FP8_TYPES_EXIST__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

/**
 * @brief Counts the number of tokens assigned to each expert.
 *
 * @param expert_ids     Device pointer to the sorted expert IDs [size_m].
 * @param expert_counts  Device pointer to the output counts [num_experts]
 * (must be pre-initialized to zero).
 * @param size_m         Total number of tokens.
 */
extern "C" __global__ void count_tokens_per_expert_kernel(
    const int32_t* expert_ids,
    int32_t* expert_counts,
    int size_m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size_m) {
        int32_t expert_id = expert_ids[i];
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

// This performs an EXCLUSIVE scan: [c0, c1] -> [0, c0, c0+c1]
// Assumptions: num_experts <= 1024 (fits in one block)
extern "C" __global__ void expert_prefix_sum_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int num_experts
) {
    extern __shared__ int32_t temp_storage[];

    int tid = threadIdx.x;

    int val = (tid < num_experts) ? counts[tid] : 0;
    temp_storage[tid] = val;

    __syncthreads();

    // Hillis-Steele Parallel Scan (Inclusive in shared mem)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int temp_val = 0;
        if (tid >= offset) {
            temp_val = temp_storage[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            temp_storage[tid] += temp_val;
        }
        __syncthreads();
    }

    if (tid < num_experts) {
        offsets[tid + 1] = temp_storage[tid];
        if (tid == 0) {
            offsets[0] = 0;
        }
    }
}

namespace vllm_rs {

static inline __device__ uint16_t float_to_half(float f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#ifndef USE_ROCM
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#else
  asm volatile("v_cvt_f16_f32 %0, %1;\n" : "=v"(tmp.u32) : "v"(f));
#endif
  return tmp.u16[0];
}

static inline __device__ void from_float(half& dst, float src) {
  dst = static_cast<half>(float_to_half(src));
}

static inline __device__ void from_float(__nv_bfloat16& dst, float src) {
  dst = __float2bfloat16(src);
}

// Non-reference overload for use in WMMA store
static inline __device__ void from_float(half* dst, float src) {
  *dst = static_cast<half>(float_to_half(src));
}

static inline __device__ void from_float(__nv_bfloat16* dst, float src) {
  *dst = __float2bfloat16(src);
}

}
