#include "cuda_fp16.h"

/*#ifndef USE_ROCM
  #define LDG(arg) __ldg(arg)
#else
  #define LDG(arg) *arg
#endif

template<typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
  scalar_t* __restrict__ arr,
  const scalar_t* __restrict__ cos_ptr,
  const scalar_t* __restrict__ sin_ptr,
  int rot_offset,
  int embed_dim)
{
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = LDG(cos_ptr + x_index);
    sin = LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = LDG(cos_ptr + x_index / 2);
    sin = LDG(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template<typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
  scalar_t* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const scalar_t* cache_ptr,
  const int head_size,
  const int num_heads,
  const int num_kv_heads,
  const int rot_dim,
  const int token_idx,
  const int64_t query_stride,
  const int64_t key_stride)
{
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(query + token_head, cos_ptr,
                                              sin_ptr, rot_offset, embed_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(key + token_head, cos_ptr,
                                              sin_ptr, rot_offset, embed_dim);
  }
}

extern "C" __global__ void rotary_embedding_kernel_f32(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  float* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  float* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const float* __restrict__ cos_sin_cache,   // [max_position, rot_dim]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const float* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<float, false>(query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim, token_idx, query_stride, key_stride);
}

extern "C" __global__ void rotary_embedding_kernel_f16(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  __half* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  __half* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const __half* __restrict__ cos_sin_cache,   // [max_position, rot_dim]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const __half* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<__half, false>(query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim, token_idx, query_stride, key_stride);
}

extern "C" __global__ void rotary_embedding_kernel_f64(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  double* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  double* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const double* __restrict__ cos_sin_cache,   // [max_position, rot_dim]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const double* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<double, false>(query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim, token_idx, query_stride, key_stride);
}




extern "C" __global__ void rotary_embedding_kernel_neox_f32(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  float* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  float* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const float* __restrict__ cos_sin_cache,   // [max_position, rot_dim]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const float* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<float, true>(query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim, token_idx, query_stride, key_stride);
}

extern "C" __global__ void rotary_embedding_kernel_neox_f16(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  __half* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  __half* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const __half* __restrict__ cos_sin_cache,   // [max_position, rot_dim]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const __half* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<__half, true>(query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim, token_idx, query_stride, key_stride);
}

extern "C" __global__ void rotary_embedding_kernel_neox_f64(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  double* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  double* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const double* __restrict__ cos_sin_cache,   // [max_position, rot_dim]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const double* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<double, true>(query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim, token_idx, query_stride, key_stride);
}

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
extern "C" __global__ void rotary_embedding_kernel_bf16(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  __nv_bfloat16* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  __nv_bfloat16* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const __nv_bfloat16* __restrict__ cos_sin_cache,   // [max_position, rot_dim]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const __nv_bfloat16* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<__nv_bfloat16, false>(query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim, token_idx, query_stride, key_stride);
}

extern "C" __global__ void rotary_embedding_kernel_neox_bf16(
  const int64_t* __restrict__ positions,        // [batch_size, seq_len] or [num_tokens]
  __nv_bfloat16* __restrict__ query,                 // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
  __nv_bfloat16* __restrict__ key,                   // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
  const __nv_bfloat16* __restrict__ cos_sin_cache,   // [max_position, rot_dim]
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const __nv_bfloat16* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<__nv_bfloat16, true>(query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim, token_idx, query_stride, key_stride);
}
#endif
*/


template <typename scalar_t_0, typename scalar_t_1>
__device__ void fused_rope_cached_forward(
    const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const int o_stride_s, const int o_stride_b, const int o_stride_h,
    const int o_stride_d, const scalar_t_0* src, const scalar_t_1* cos,
    const scalar_t_1* sin, scalar_t_0* dst, int64_t* positions) {
  int b_id = blockIdx.y;
  int s_id = blockIdx.x;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    scalar_t_0 v_cos = cos[s_id * d2 + d_id];
    scalar_t_0 v_sin = sin[s_id * d2 + d_id];
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      scalar_t_0 v_src = src[offset_src];
      scalar_t_0 v_src_rotate =
          (d_id + d2 / 2 < d2) ? -src[offset_src + (d2 / 2) * stride_d]
                               : src[offset_src + (d2 / 2 - d2) * stride_d];
      dst[offset_dst] = 1;//v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

extern "C" __global__ void rotary_embedding_kernel_f32(
    const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const int o_stride_s, const int o_stride_b, const int o_stride_h,
    const int o_stride_d, const float* src, const float* cos,
    const float* sin, float* dst, int64_t* positions) {
    fused_rope_cached_forward(
      h, d, d2,
      stride_s, stride_b, stride_h, stride_d,
      o_stride_s, o_stride_b, o_stride_h, o_stride_d, 
      src, cos, sin, dst,
      positions);
}

extern "C" __global__ void rotary_embedding_kernel_f64(
    const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const int o_stride_s, const int o_stride_b, const int o_stride_h,
    const int o_stride_d, const double* src, const float* cos,
    const float* sin, double* dst, int64_t* positions) {
    fused_rope_cached_forward(
      h, d, d2,
      stride_s, stride_b, stride_h, stride_d,
      o_stride_s, o_stride_b, o_stride_h, o_stride_d, 
      src, cos, sin, dst,
      positions);
}

extern "C" __global__ void rotary_embedding_kernel_f16(
    const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const int o_stride_s, const int o_stride_b, const int o_stride_h,
    const int o_stride_d, const __half* src, const float* cos,
    const float* sin, __half* dst, int64_t* positions) {
    fused_rope_cached_forward(
      h, d, d2,
      stride_s, stride_b, stride_h, stride_d,
      o_stride_s, o_stride_b, o_stride_h, o_stride_d, 
      src, cos, sin, dst,
      positions);
}

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
extern "C" __global__ void rotary_embedding_kernel_bf16(
    const int h, const int d, const int d2, const int stride_s,
    const int stride_b, const int stride_h, const int stride_d,
    const int o_stride_s, const int o_stride_b, const int o_stride_h,
    const int o_stride_d, const __nv_bfloat16* src, const float* cos,
    const float* sin, __nv_bfloat16* dst, int64_t* positions) {
    fused_rope_cached_forward(
      h, d, d2,
      stride_s, stride_b, stride_h, stride_d,
      o_stride_s, o_stride_b, o_stride_h, o_stride_d, 
      src, cos, sin, dst,
      positions);
}
#endif
