// Heterogeneous multi-LoRA batching kernels (S-LoRA / Punica style BGMV).
//
// The batched LoRA delta `delta[i] = x[i] · A_stack[slot(i)] · B_stack[slot(i)]`
// is computed in two gather-GEMV passes, exactly matching the reference
// gather + batched-matmul path in candle-nn (`batched_lora_delta`):
//
//   shrink:  tmp[i, :]   = x[i, :]   · A_stack[slot(i)]   (in  -> r)
//   expand:  delta[i, :] = tmp[i, :] · B_stack[slot(i)]   (r   -> out)
//
// `A_stack` holds `A^T` per slot (rank-padded to `r` with zeros); `B_stack`
// holds `B^T` per slot with the `alpha/r` scaling already folded in and the
// same rank padding. Slot 0 is an all-zero adapter used by the rows that
// select no adapter, so those rows produce a zero delta. Reading the stacks
// in place through `slots` avoids materializing the per-row gather of `A`/`B`
// that the reference path builds before its matmuls.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define LORA_DTYPE_F32 0
#define LORA_DTYPE_F16 1
#define LORA_DTYPE_BF16 2

template <typename T> __device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// tmp[i, j] = sum_k x[i, k] * A_stack[slot(i), k, j]
// x:   [batch, in_dim]            (row-major, contiguous)
// A:   [n_slots, in_dim, r]       (row-major, contiguous)
// tmp: [batch, r]
// slots: [batch]                  (u32, one slot index per row)
template <typename T>
__global__ void lora_bgmv_shrink_kernel(const T *__restrict__ x, const T *__restrict__ a,
                                        const uint32_t *__restrict__ slots, T *__restrict__ tmp,
                                        uint32_t batch, uint32_t in_dim, uint32_t r) {
  uint32_t i = blockIdx.x;
  uint32_t j = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= batch || j >= r) {
    return;
  }
  uint32_t slot = slots[i];
  const T *xrow = x + (size_t)i * in_dim;
  const T *aslot = a + ((size_t)slot * in_dim) * r; // A_stack[slot], layout [in_dim, r]
  float acc = 0.f;
  for (uint32_t k = 0; k < in_dim; ++k) {
    acc += to_f32<T>(xrow[k]) * to_f32<T>(aslot[(size_t)k * r + j]);
  }
  tmp[(size_t)i * r + j] = from_f32<T>(acc);
}

// delta[i, o] = sum_j tmp[i, j] * B_stack[slot(i), j, o]
// tmp:   [batch, r]
// B:     [n_slots, r, out_dim]    (row-major, contiguous)
// delta: [batch, out_dim]
// slots: [batch]
template <typename T>
__global__ void lora_bgmv_expand_kernel(const T *__restrict__ tmp, const T *__restrict__ b,
                                        const uint32_t *__restrict__ slots, T *__restrict__ delta,
                                        uint32_t batch, uint32_t r, uint32_t out_dim) {
  uint32_t i = blockIdx.x;
  uint32_t o = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= batch || o >= out_dim) {
    return;
  }
  uint32_t slot = slots[i];
  const T *trow = tmp + (size_t)i * r;
  const T *bslot = b + ((size_t)slot * r) * out_dim; // B_stack[slot], layout [r, out_dim]
  float acc = 0.f;
  for (uint32_t j = 0; j < r; ++j) {
    acc += to_f32<T>(trow[j]) * to_f32<T>(bslot[(size_t)j * out_dim + o]);
  }
  delta[(size_t)i * out_dim + o] = from_f32<T>(acc);
}

#define BLOCK 256

extern "C" void lora_bgmv_shrink(const void *x, const void *a, const uint32_t *slots, void *tmp,
                                 uint32_t batch, uint32_t in_dim, uint32_t r, uint32_t dtype,
                                 void *stream) {
  if (batch == 0 || r == 0) {
    return;
  }
  cudaStream_t s = (cudaStream_t)stream;
  dim3 grid(batch, (r + BLOCK - 1) / BLOCK);
  dim3 block(BLOCK);
  switch (dtype) {
  case LORA_DTYPE_F32:
    lora_bgmv_shrink_kernel<float><<<grid, block, 0, s>>>(
        (const float *)x, (const float *)a, slots, (float *)tmp, batch, in_dim, r);
    break;
  case LORA_DTYPE_F16:
    lora_bgmv_shrink_kernel<__half><<<grid, block, 0, s>>>(
        (const __half *)x, (const __half *)a, slots, (__half *)tmp, batch, in_dim, r);
    break;
  case LORA_DTYPE_BF16:
    lora_bgmv_shrink_kernel<__nv_bfloat16><<<grid, block, 0, s>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)a, slots, (__nv_bfloat16 *)tmp, batch,
        in_dim, r);
    break;
  }
}

extern "C" void lora_bgmv_expand(const void *tmp, const void *b, const uint32_t *slots, void *delta,
                                 uint32_t batch, uint32_t r, uint32_t out_dim, uint32_t dtype,
                                 void *stream) {
  if (batch == 0 || out_dim == 0) {
    return;
  }
  cudaStream_t s = (cudaStream_t)stream;
  dim3 grid(batch, (out_dim + BLOCK - 1) / BLOCK);
  dim3 block(BLOCK);
  switch (dtype) {
  case LORA_DTYPE_F32:
    lora_bgmv_expand_kernel<float><<<grid, block, 0, s>>>(
        (const float *)tmp, (const float *)b, slots, (float *)delta, batch, r, out_dim);
    break;
  case LORA_DTYPE_F16:
    lora_bgmv_expand_kernel<__half><<<grid, block, 0, s>>>(
        (const __half *)tmp, (const __half *)b, slots, (__half *)delta, batch, r, out_dim);
    break;
  case LORA_DTYPE_BF16:
    lora_bgmv_expand_kernel<__nv_bfloat16><<<grid, block, 0, s>>>(
        (const __nv_bfloat16 *)tmp, (const __nv_bfloat16 *)b, slots, (__nv_bfloat16 *)delta, batch,
        r, out_dim);
    break;
  }
}
