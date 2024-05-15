#include "cuda_fp16.h"
#include <stdint.h>

#define WARP_SIZE 32

#ifndef USE_ROCM
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#else
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#endif

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1)
    val += VLLM_SHFL_XOR_SYNC(val, mask);
  return val;
}

__inline__ __device__ constexpr int _calculateLaneMask(int warp_size) {
  return warp_size - 1;
}

__inline__ __device__ constexpr int _calculateWidShift(int warp_size) {
  return 5 + (warp_size >> 6);
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[WARP_SIZE];
  constexpr auto LANE_MASK = _calculateLaneMask(WARP_SIZE);
  constexpr auto WID_SHIFT = _calculateWidShift(WARP_SIZE);
  int lane = threadIdx.x & LANE_MASK;
  int wid = threadIdx.x >> WID_SHIFT;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / (WARP_SIZE * 1.0f))) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}

#define RMS_NORM_OP(FN_NAME, TYPENAME)\
extern "C" __global__ void FN_NAME(\
  TYPENAME* __restrict__ out,\
  const TYPENAME* __restrict__ input,\
  const TYPENAME* __restrict__ weight,\
  const float epsilon,\
  const int num_tokens,\
  const int hidden_size) {\
  __shared__ float s_variance;\
  float variance = 0.0f;\
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {\
    const float x = (float) input[blockIdx.x * hidden_size + idx];\
    variance += x * x;\
  }\
  variance = blockReduceSum<float>(variance);\
  if (threadIdx.x == 0) {\
    s_variance = rsqrtf(variance / hidden_size + epsilon);\
  }\
  __syncthreads();\
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {\
    float x = (float) input[blockIdx.x * hidden_size + idx];\
    out[blockIdx.x * hidden_size + idx] = ((TYPENAME) (x * s_variance)) * weight[idx];\
  }\
}\

RMS_NORM_OP(rms_norm_f32, float)
RMS_NORM_OP(rms_norm_f16, __half)

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
RMS_NORM_OP(rms_norm_bf16, __nv_bfloat16)
#endif