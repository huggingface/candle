// Single-token (decode) attention kernel: each sequence in the batch contributes exactly one
// new query token, which attends over its full key/value cache. This is the computational
// pattern FlashInfer's batch-decode kernels target, and is the workload referenced by
// https://github.com/huggingface/candle/issues/3651. The kernel below is a straightforward,
// numerically-stable (streaming softmax) reference implementation: one CUDA block per
// (batch, query head), one thread per head-dimension element, looping sequentially over the
// key/value cache. It is not a tensor-core / split-KV kernel, so it should not be expected to
// match FlashInfer's own throughput, but it gives candle a working, feature-gated decode-attention
// backend with the same shape contract.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

template <typename T>
__device__ __forceinline__ float to_f32(T x);
template <>
__device__ __forceinline__ float to_f32<float>(float x) {
  return x;
}
template <>
__device__ __forceinline__ float to_f32<__half>(__half x) {
  return __half2float(x);
}
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_f32(float x);
template <>
__device__ __forceinline__ float from_f32<float>(float x) {
  return x;
}
template <>
__device__ __forceinline__ __half from_f32<__half>(float x) {
  return __float2half(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

// grid = (batch, num_query_heads), block = next_pow2(head_dim) threads (<= 1024).
template <typename T>
__global__ void decode_attention_kernel(
    const T *__restrict__ q, const T *__restrict__ k, const T *__restrict__ v,
    T *__restrict__ out, int hkv_group, int seqlen_k, int head_dim,
    int64_t q_b_stride, int64_t q_h_stride, int64_t k_b_stride,
    int64_t k_h_stride, int64_t k_l_stride, int64_t v_b_stride,
    int64_t v_h_stride, int64_t v_l_stride, int64_t o_b_stride,
    int64_t o_h_stride, float scale) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int hkv = h / hkv_group;
  const int tid = threadIdx.x;

  extern __shared__ float red[];

  const T *q_ptr = q + (int64_t)b * q_b_stride + (int64_t)h * q_h_stride;
  const T *k_base = k + (int64_t)b * k_b_stride + (int64_t)hkv * k_h_stride;
  const T *v_base = v + (int64_t)b * v_b_stride + (int64_t)hkv * v_h_stride;
  T *o_ptr = out + (int64_t)b * o_b_stride + (int64_t)h * o_h_stride;

  const float qd = (tid < head_dim) ? to_f32<T>(q_ptr[tid]) : 0.f;

  float m = -INFINITY;
  float l = 0.f;
  float acc = 0.f;

  for (int t = 0; t < seqlen_k; ++t) {
    const T *k_row = k_base + (int64_t)t * k_l_stride;
    red[tid] = (tid < head_dim) ? qd * to_f32<T>(k_row[tid]) : 0.f;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (tid < offset) {
        red[tid] += red[tid + offset];
      }
      __syncthreads();
    }
    const float score = red[0] * scale;
    __syncthreads();

    const float new_m = fmaxf(m, score);
    const float corr = __expf(m - new_m);
    const float p = __expf(score - new_m);
    l = l * corr + p;

    const T *v_row = v_base + (int64_t)t * v_l_stride;
    const float vd = (tid < head_dim) ? to_f32<T>(v_row[tid]) : 0.f;
    acc = acc * corr + p * vd;
    m = new_m;
  }

  if (tid < head_dim) {
    o_ptr[tid] = from_f32<T>(acc / l);
  }
}

template <typename T>
static void launch(const T *q, const T *k, const T *v, T *out, int batch,
                    int num_heads, int num_heads_k, int seqlen_k,
                    int head_dim, int64_t q_b_stride, int64_t q_h_stride,
                    int64_t k_b_stride, int64_t k_h_stride,
                    int64_t k_l_stride, int64_t v_b_stride,
                    int64_t v_h_stride, int64_t v_l_stride,
                    int64_t o_b_stride, int64_t o_h_stride, float scale,
                    cudaStream_t stream) {
  int block = 1;
  while (block < head_dim) {
    block <<= 1;
  }
  if (block > 1024) {
    block = 1024;
  }
  const dim3 grid(batch, num_heads);
  const size_t smem = (size_t)block * sizeof(float);
  decode_attention_kernel<T><<<grid, block, smem, stream>>>(
      q, k, v, out, num_heads / num_heads_k, seqlen_k, head_dim, q_b_stride,
      q_h_stride, k_b_stride, k_h_stride, k_l_stride, v_b_stride, v_h_stride,
      v_l_stride, o_b_stride, o_h_stride, scale);
}

extern "C" void run_decode_attention(
    const void *q, const void *k, const void *v, void *out, int32_t dtype,
    int32_t batch, int32_t num_heads, int32_t num_heads_k, int32_t seqlen_k,
    int32_t head_dim, int64_t q_b_stride, int64_t q_h_stride,
    int64_t k_b_stride, int64_t k_h_stride, int64_t k_l_stride,
    int64_t v_b_stride, int64_t v_h_stride, int64_t v_l_stride,
    int64_t o_b_stride, int64_t o_h_stride, float scale,
    cudaStream_t stream) {
  switch (dtype) {
  case 0:
    launch<float>((const float *)q, (const float *)k, (const float *)v,
                  (float *)out, batch, num_heads, num_heads_k, seqlen_k,
                  head_dim, q_b_stride, q_h_stride, k_b_stride, k_h_stride,
                  k_l_stride, v_b_stride, v_h_stride, v_l_stride, o_b_stride,
                  o_h_stride, scale, stream);
    break;
  case 1:
    launch<__half>((const __half *)q, (const __half *)k, (const __half *)v,
                    (__half *)out, batch, num_heads, num_heads_k, seqlen_k,
                    head_dim, q_b_stride, q_h_stride, k_b_stride, k_h_stride,
                    k_l_stride, v_b_stride, v_h_stride, v_l_stride,
                    o_b_stride, o_h_stride, scale, stream);
    break;
  default:
    launch<__nv_bfloat16>((const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k,
                           (const __nv_bfloat16 *)v, (__nv_bfloat16 *)out,
                           batch, num_heads, num_heads_k, seqlen_k, head_dim,
                           q_b_stride, q_h_stride, k_b_stride, k_h_stride,
                           k_l_stride, v_b_stride, v_h_stride, v_l_stride,
                           o_b_stride, o_h_stride, scale, stream);
    break;
  }
}
