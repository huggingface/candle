#include "cuda_fp16.h"

template <typename scalar_t_0, typename scalar_t_1>
__device__ void fused_rope_cached_forward(
    const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const scalar_t_0* src, const scalar_t_1* cos,
    const scalar_t_1* sin, scalar_t_0* dst, int64_t* positions) {
  int b_id = blockIdx.y;
  int s_id = blockIdx.x;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * stride_s + b_id * stride_b;
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    scalar_t_0 v_cos = cos[(s_id + positions[b_id]) * d2 + d_id];
    scalar_t_0 v_sin = sin[(s_id + positions[b_id]) * d2 + d_id];
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * stride_h + d_id * stride_d;
      scalar_t_0 v_src = src[offset_src];
      scalar_t_0 v_src_rotate =
          (d_id + d2 / 2 < d2) ? -src[offset_src + (d2 / 2) * stride_d]
                               : src[offset_src + (d2 / 2 - d2) * stride_d];
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

extern "C" __global__ void rotary_embedding_kernel_f32(
    const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const float* src, const float* cos,
    const float* sin, float* dst, int64_t* positions) {
    fused_rope_cached_forward(
      h, d, d2,
      stride_s, stride_b, stride_h, stride_d,
      src, cos, sin, dst,
      positions);
}

extern "C" __global__ void rotary_embedding_kernel_f64(
    const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const double* src, const float* cos,
    const float* sin, double* dst, int64_t* positions) {
    fused_rope_cached_forward(
      h, d, d2,
      stride_s, stride_b, stride_h, stride_d,
      src, cos, sin, dst,
      positions);
}

extern "C" __global__ void rotary_embedding_kernel_f16(
    const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const __half* src, const float* cos,
    const float* sin, __half* dst, int64_t* positions) {
    fused_rope_cached_forward(
      h, d, d2,
      stride_s, stride_b, stride_h, stride_d,
      src, cos, sin, dst,
      positions);
}

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
extern "C" __global__ void rotary_embedding_kernel_bf16(
    const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const __nv_bfloat16* src, const float* cos,
    const float* sin, __nv_bfloat16* dst, int64_t* positions) {
    fused_rope_cached_forward(
      h, d, d2,
      stride_s, stride_b, stride_h, stride_d,
      src, cos, sin, dst,
      positions);
}
#endif
