#include "cuda_utils.cuh"
#include <cmath>
#include <stdint.h>
#include <cuda/std/limits>

#define WARP_SIZE 32
const int BLOCK_SIZE = 1024;

// Helpers to initialize reduction identities for both floating-point and
// integer types. For floats we keep using +/-INFINITY, while for integers
// we use well-defined numeric_limits values instead of relying on casting
// +/-INFINITY to an integer type (which is undefined behaviour and has been
// observed to break on newer GPU architectures such as Blackwell).
template <typename T>
__device__ __forceinline__ T reduce_init_lowest() {
  // Default implementation is used for floating-point types (__half,
  // __nv_bfloat16, float, double). The conversion from -INFINITY (double)
  // to these types is well-defined and produces -inf.
  return -INFINITY;
}

template <typename T>
__device__ __forceinline__ T reduce_init_highest() {
  // Default implementation is used for floating-point types (__half,
  // __nv_bfloat16, float, double). The conversion from INFINITY (double)
  // to these types is well-defined and produces +inf.
  return INFINITY;
}

// Integer specializations – use numeric_limits instead of +/-INFINITY.
template <>
__device__ __forceinline__ int64_t reduce_init_lowest<int64_t>() {
  return ::cuda::std::numeric_limits<int64_t>::lowest();
}

template <>
__device__ __forceinline__ uint32_t reduce_init_lowest<uint32_t>() {
  return ::cuda::std::numeric_limits<uint32_t>::lowest();
}

template <>
__device__ __forceinline__ uint8_t reduce_init_lowest<uint8_t>() {
  return ::cuda::std::numeric_limits<uint8_t>::lowest();
}

template <>
__device__ __forceinline__ int64_t reduce_init_highest<int64_t>() {
  return ::cuda::std::numeric_limits<int64_t>::max();
}

template <>
__device__ __forceinline__ uint32_t reduce_init_highest<uint32_t>() {
  return ::cuda::std::numeric_limits<uint32_t>::max();
}

template <>
__device__ __forceinline__ uint8_t reduce_init_highest<uint8_t>() {
  return ::cuda::std::numeric_limits<uint8_t>::max();
}

// TODO: Maybe add some fast_sum_f16_f32 variant that not only accumulate in f32
// but also expect a f32 output so that this can be used for normalization e.g.
// in softmax.

// Optimized reduce sum: contiguous fast path with vectorized loads + warp shuffle,
// falls back to strided path for non-contiguous data.
template <typename T>
__device__ void
fast_sum(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, T *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ float shr[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);

  float local_sum = 0.0f;

  if (is_contiguous(num_dims, dims, strides)) {
    size_t idx = start_idx + tid;
    while (idx < stop_idx) {
      local_sum += static_cast<float>(src[idx]);
      idx += blockDim.x;
    }
  } else {
    size_t idx = start_idx + tid;
    while (idx < stop_idx) {
      size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
      local_sum += static_cast<float>(src[strided_i]);
      idx += blockDim.x;
    }
  }

  // Warp-level reduction first
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  if (lane_id == 0) shr[warp_id] = local_sum;
  __syncthreads();

  // Final reduction across warps
  int num_warps = blockDim.x / WARP_SIZE;
  if (tid < WARP_SIZE) {
    local_sum = (tid < num_warps) ? shr[tid] : 0.0f;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
      local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if (tid == 0) dst[dst_id] = static_cast<T>(local_sum);
  }
}

// Specialized vectorized fast_sum for bf16: 8 elements per float4 load
#if __CUDA_ARCH__ >= 800
__device__ void
fast_sum_bf16_vec(const size_t src_numel, const size_t el_to_sum_per_block,
                  const size_t num_dims, const size_t *info,
                  const __nv_bfloat16 *src, __nv_bfloat16 *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);

  float local_sum = 0.0f;

  if (is_contiguous(num_dims, dims, strides)) {
    // Round start up and stop down to float4-aligned boundaries
    const size_t vec_start = (start_idx + 7) / 8;
    const size_t vec_stop = stop_idx / 8;

    // Scalar head: elements before the first aligned float4
    size_t head_end = min(vec_start * 8, stop_idx);
    for (size_t i = start_idx + tid; i < head_end; i += blockDim.x) {
      local_sum += __bfloat162float(src[i]);
    }

    if (vec_start < vec_stop && is_aligned_16(src)) {
      const float4 *src4 = reinterpret_cast<const float4*>(src);
      for (size_t vi = vec_start + tid; vi < vec_stop; vi += blockDim.x) {
        float4 v = src4[vi];
        const __nv_bfloat16 *bp = reinterpret_cast<const __nv_bfloat16*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
          local_sum += __bfloat162float(bp[j]);
        }
      }
    } else {
      for (size_t i = head_end + tid; i < vec_stop * 8; i += blockDim.x) {
        local_sum += __bfloat162float(src[i]);
      }
    }

    // Scalar tail: elements after the last aligned float4
    size_t tail_start = vec_stop * 8;
    for (size_t i = tail_start + tid; i < stop_idx; i += blockDim.x) {
      local_sum += __bfloat162float(src[i]);
    }
  } else {
    size_t idx = start_idx + tid;
    while (idx < stop_idx) {
      size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
      local_sum += __bfloat162float(src[strided_i]);
      idx += blockDim.x;
    }
  }

  // Warp-level reduction
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);

  __shared__ float shr[32];
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  if (lane_id == 0) shr[warp_id] = local_sum;
  __syncthreads();

  int num_warps = blockDim.x / WARP_SIZE;
  if (tid < WARP_SIZE) {
    local_sum = (tid < num_warps) ? shr[tid] : 0.0f;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
      local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if (tid == 0) dst[dst_id] = __float2bfloat16(local_sum);
  }
}
#endif

static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, mask, 32);
        a.y += __shfl_xor_sync(0xffffffff, a.y, mask, 32);
    }
    return a;
}

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

// LayerNorm implementation adapted from ggml, accumulation is made using f32.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L477
template <typename T>
__device__ void layernorm(const T * x, T * dst, const T * alpha, const T * beta, const int ncols, const int block_size, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float2 mean_var = make_float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var);
    if (block_size > WARP_SIZE) {
        __shared__ float2 s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        __syncthreads();
        mean_var = s_sum[lane_id];
        mean_var = warp_reduce_sum(mean_var);
    }

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    if (alpha == nullptr && beta == nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float lhs = (static_cast<float>(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = static_cast<T>(lhs);
      }
    }
    else if (alpha == nullptr && beta != nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float b = static_cast<float>(beta[col]);
          float lhs = (static_cast<float>(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = static_cast<T>(lhs + b);
      }
    }
    else if (alpha != nullptr && beta == nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float a = static_cast<float>(alpha[col]);
          float lhs = (static_cast<float>(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = static_cast<T>(lhs * a);
      }
    }
    else {
      for (int col = tid; col < ncols; col += block_size) {
          float a = static_cast<float>(alpha[col]);
          float b = static_cast<float>(beta[col]);
          float lhs = (static_cast<float>(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = static_cast<T>(lhs * a + b);
      }
    }
}

// RmsNorm implementation adapted from ggml, accumulation is made using f32.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L523
template <typename T>
__device__ void rmsnorm(const T * x, T * dst, const T * alpha, const int ncols, const int block_size, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = static_cast<float>(x[row*ncols + col]);
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if (alpha == nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          dst[row*ncols + col] = static_cast<T>(scale * static_cast<float>(x[row*ncols + col]));
      }
    }
    else {
      for (int col = tid; col < ncols; col += block_size) {
          float a = static_cast<float>(alpha[col]);
          dst[row*ncols + col] = static_cast<T>(scale * static_cast<float>(x[row*ncols + col]) * a);
      }
    }
}

// Softmax implementation adapted from ggml.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L4159
template <typename T, typename ACC>
__device__ void softmax(const T * x, T * dst, const int ncols) {
    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;

    T max_val = -INFINITY;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        max_val = maxg(max_val, x[i]);
    }

    // find the max value in the block
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        max_val = maxg(max_val, __shfl_xor_sync(0xffffffff, max_val, mask, 32));
    }

    ACC tmp = 0.;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        const T val = expg(x[i] - max_val);
        tmp += static_cast<ACC>(val);
        dst[i] = val;
    }

    // sum up partial sums
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    const ACC inv_tmp = 1. / tmp;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        dst[i] *= inv_tmp;
    }
}

template <typename T>
__device__ void ropei(const T * src, const T * cos, const T * sin, T * dst, const uint32_t bh, const uint32_t td, const uint32_t stride_b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= bh * td) return;

    uint32_t rope_idx = idx % (td / 2);
    if (stride_b > 0) {
      uint32_t b_idx = (2 * idx) / stride_b;
      rope_idx += b_idx * (td / 2);
    }
    T c = cos[rope_idx];
    T s = sin[rope_idx];

    dst[2 * idx] = src[2 * idx] * c - src[2 * idx + 1] * s;
    dst[2 * idx + 1] = src[2 * idx] * s + src[2 * idx + 1] * c;
}

template <typename T>
__device__ void rope(const T * src, const T * cos, const T * sin, T * dst, const uint32_t bh, const uint32_t td, const uint32_t d, const uint32_t stride_b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= bh * td) return;

    uint32_t i_bh = idx / (td / 2);
    uint32_t i_td = idx - (td / 2) * i_bh;
    uint32_t i_t = i_td / (d / 2);
    uint32_t i_d = i_td - (d / 2) * i_t;
    uint32_t i1 = i_bh * td + i_t * d + i_d;
    uint32_t i2 = i1 + d / 2;
    uint32_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
      uint32_t b_idx = (2 * idx) / stride_b;
      i_cs += b_idx * (td / 2);
    }
    T c = cos[i_cs];
    T s = sin[i_cs];

    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template <typename T>
__device__ void rope_thd(
    const T * src,
    const T * cos,
    const T * sin,
    T * dst,
    const uint32_t b,
    const uint32_t t,
    const uint32_t h,
    const uint32_t d,
    const uint32_t stride_b
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= b * t * h * d) return;

    uint32_t i_bth = idx / (d / 2);
    uint32_t i_d = idx - (d / 2) * i_bth;
    uint32_t i_t = (i_bth / h) % t;
    uint32_t i1 = i_bth * d + i_d;
    uint32_t i2 = i1 + d / 2;
    uint32_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
      uint32_t b_idx = (2 * idx) / stride_b;
      i_cs += b_idx * ((t * d) / 2);
    }
    T c = cos[i_cs];
    T s = sin[i_cs];

    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template <typename T>
__device__ void
fast_max(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, T *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  // Initialize with the lowest representable value for T so that the first
  // comparison in the reduction always picks a real element.
  shr[tid] = reduce_init_lowest<T>();
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    shr[tid] = maxg(shr[tid], src[strided_i]);
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s)
      shr[tid] = maxg(shr[tid], shr[tid + s]);
  }

  if (tid == 0)
    dst[dst_id] = shr[0];
}

template <typename T>
__device__ void
fast_min(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, T *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  // Initialize with the highest representable value for T so that the first
  // comparison in the reduction always picks a real element.
  shr[tid] = reduce_init_highest<T>();
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    shr[tid] = ming(shr[tid], src[strided_i]);
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s)
      shr[tid] = ming(shr[tid], shr[tid + s]);
  }

  if (tid == 0)
    dst[dst_id] = shr[0];
}

template <typename T>
__device__ void
fast_argmin(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, uint32_t *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  __shared__ uint32_t shr_index[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  // For floating types this uses +inf; for integer types we use the largest
  // representable value instead of casting INFINITY to an integer.
  shr[tid] = reduce_init_highest<T>();
  shr_index[tid] = 0xFFFFFFFF;
  bool not_set = true;
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    if (not_set || src[strided_i] < shr[tid]) {
      shr[tid] = src[strided_i];
      // Assume that the reduction takes place over the last dimension which is contiguous.
      shr_index[tid] = idx % dims[num_dims - 1];
      not_set = false;
    }
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s && shr[tid + s] < shr[tid]) {
      shr[tid] = shr[tid + s];
      shr_index[tid] = shr_index[tid + s];
    }
  }

  if (tid == 0)
    dst[dst_id] = shr_index[0];
}

template <typename T>
__device__ void
fast_argmax(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, uint32_t *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  __shared__ uint32_t shr_index[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  // For floating types this uses -inf; for integer types we use the lowest
  // representable value instead of casting -INFINITY to an integer.
  shr[tid] = reduce_init_lowest<T>();
  shr_index[tid] = 0xFFFFFFFF;
  bool not_set = true;
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    if (not_set || src[strided_i] > shr[tid]) {
      shr[tid] = src[strided_i];
      // Assume that the reduction takes place over the last dimension which is contiguous.
      shr_index[tid] = idx % dims[num_dims - 1];
      not_set = false;
    }
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s && shr[tid + s] > shr[tid]) {
      shr[tid] = shr[tid + s];
      shr_index[tid] = shr_index[tid + s];
    }
  }

  if (tid == 0)
    dst[dst_id] = shr_index[0];
}

#define FAST_OP(TYPENAME, MIN_NAME, MAX_NAME, ARGMIN_NAME, ARGMAX_NAME, SUM_NAME) \
  extern "C" __global__ void ARGMIN_NAME(                                      \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      uint32_t *dst) {                                                         \
    fast_argmin(src_numel, el_to_sum_per_block, num_dims, info, src, dst);     \
  }                                                                            \
  extern "C" __global__ void ARGMAX_NAME(                                     \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      uint32_t *dst) {                                                         \
    fast_argmax(src_numel, el_to_sum_per_block, num_dims, info, src, dst);     \
  }                                                                            \
  extern "C" __global__ void MIN_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_min(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }                                                                            \
  extern "C" __global__ void MAX_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_max(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }                                                                            \
  extern "C" __global__ void SUM_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_sum(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }

#define SUM_OP(TYPENAME, FN_NAME)                                              \
  extern "C" __global__ void FN_NAME(                                          \
      const size_t numel, const size_t num_dims, const size_t num_sum_dims,    \
      const size_t *info, const TYPENAME *inp, TYPENAME *out) {                \
    const size_t *dims = info;                                                 \
    const size_t *strides = info + num_dims;                                   \
    const size_t *sum_dims_l = info + 2 * num_dims;                            \
    const size_t *sum_dims_s = info + 2 * num_dims + num_sum_dims;             \
    if (is_contiguous(num_dims, dims, strides)) {                              \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        size_t dst_index = i;                                                  \
        for (unsigned int nd = 0; nd < num_sum_dims; ++nd) {                   \
          size_t stride = sum_dims_s[nd];                                      \
          size_t pre = dst_index / stride;                                     \
          size_t post = dst_index % stride;                                    \
          dst_index = (pre / sum_dims_l[nd]) * stride + post;                  \
        }                                                                      \
        atomicAdd(out + dst_index, inp[i]);                                    \
      }                                                                        \
    } else {                                                                   \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        unsigned strided_i = get_strided_index(i, num_dims, dims, strides);    \
        size_t dst_index = i;                                                  \
        for (unsigned int nd = 0; nd < num_sum_dims; ++nd) {                   \
          size_t stride = sum_dims_s[nd];                                      \
          size_t pre = dst_index / stride;                                     \
          size_t post = dst_index % stride;                                    \
          dst_index = (pre / sum_dims_l[nd]) * stride + post;                  \
        }                                                                      \
        atomicAdd(out + dst_index, inp[strided_i]);                            \
      }                                                                        \
    }                                                                          \
  }

#define SOFTMAX_OP(TYPENAME, ACC_TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, TYPENAME *dst,                                      \
      const int n_cols) {                                                      \
    softmax<TYPENAME, ACC_TYPENAME>(src, dst, n_cols);                         \
  }                                                                            \

#define RMSNORM_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, TYPENAME *dst, const TYPENAME *alpha,               \
      const int n_cols, const int block_size, const float eps) {               \
    rmsnorm<TYPENAME>(src, dst, alpha, n_cols, block_size, eps);               \
  }                                                                            \

#define LAYERNORM_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, TYPENAME *dst, const TYPENAME *alpha,               \
      const TYPENAME *beta, const int n_cols, const int block_size, const float eps) { \
    layernorm<TYPENAME>(src, dst, alpha, beta, n_cols, block_size, eps);       \
  }                                                                            \

#define ROPE_OP(TYPENAME, FN_NAME, FN_NAME_I, FN_NAME_THD) \
  extern "C" __global__ void FN_NAME_I( \
      const TYPENAME *src, \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      TYPENAME *dst, \
      const uint32_t bh, \
      const uint32_t td, \
      const uint32_t stride_b) { \
    ropei<TYPENAME>(src, cos, sin, dst, bh, td, stride_b); \
  } \
  extern "C" __global__ void FN_NAME( \
      const TYPENAME *src, \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      TYPENAME *dst, \
      const uint32_t bh, \
      const uint32_t td, \
      const uint32_t d, \
      const uint32_t stride_b) { \
    rope<TYPENAME>(src, cos, sin, dst, bh, td, d, stride_b); \
  } \
  extern "C" __global__ void FN_NAME_THD( \
      const TYPENAME *src, \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      TYPENAME *dst, \
      const uint32_t b, \
      const uint32_t t, \
      const uint32_t h, \
      const uint32_t d, \
      const uint32_t stride_b) { \
    rope_thd<TYPENAME>(src, cos, sin, dst, b, t, h, d, stride_b); \
  } \

// Small-reduce kernel: one thread per output element, for el_to_sum <= 32.
// Eliminates massive block-count overhead when reducing over tiny dimensions (e.g., MoE topk=8).
template <typename T>
__device__ void
fast_sum_small_impl(const size_t src_numel, const size_t el_to_sum_per_block,
                    const size_t num_dims, const size_t *info, const T *src, T *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;
  const size_t dst_el = src_numel / el_to_sum_per_block;
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= dst_el) return;

  float sum = 0.0f;

  if (is_contiguous(num_dims, dims, strides)) {
    size_t start = gid * el_to_sum_per_block;
    for (size_t i = 0; i < el_to_sum_per_block; ++i) {
      sum += static_cast<float>(src[start + i]);
    }
  } else {
    size_t base_linear = gid * el_to_sum_per_block;
    size_t base_strided = get_strided_index(base_linear, num_dims, dims, strides);
    size_t sum_stride = strides[num_dims - 1];
    for (size_t i = 0; i < el_to_sum_per_block; ++i) {
      sum += static_cast<float>(src[base_strided + i * sum_stride]);
    }
  }
  dst[gid] = static_cast<T>(sum);
}

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void fast_sum_small_bf16(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __nv_bfloat16 *src,
    __nv_bfloat16 *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;
  const size_t dst_el = src_numel / el_to_sum_per_block;
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= dst_el) return;

  float sum = 0.0f;

  if (is_contiguous(num_dims, dims, strides)) {
    size_t start = gid * el_to_sum_per_block;
    for (size_t i = 0; i < el_to_sum_per_block; ++i) {
      sum += __bfloat162float(src[start + i]);
    }
  } else {
    // Compute base address once via get_strided_index, then use the stride
    // of the innermost (sum) dimension for subsequent elements.
    // This avoids expensive integer division per element.
    size_t base_linear = gid * el_to_sum_per_block;
    size_t base_strided = get_strided_index(base_linear, num_dims, dims, strides);
    size_t sum_stride = strides[num_dims - 1];

    for (size_t i = 0; i < el_to_sum_per_block; ++i) {
      sum += __bfloat162float(src[base_strided + i * sum_stride]);
    }
  }

  dst[gid] = __float2bfloat16(sum);
}
#endif

extern "C" __global__ void fast_sum_small_f32(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const float *src,
    float *dst) {
  fast_sum_small_impl(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}

extern "C" __global__ void fast_sum_small_f64(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const double *src,
    double *dst) {
  fast_sum_small_impl(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void fast_sum_small_f16(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __half *src,
    __half *dst) {
  fast_sum_small_impl(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}
#endif

#if __CUDA_ARCH__ >= 800
SOFTMAX_OP(__nv_bfloat16, float, softmax_bf16)
RMSNORM_OP(__nv_bfloat16, rmsnorm_bf16)
LAYERNORM_OP(__nv_bfloat16, layernorm_bf16)
ROPE_OP(__nv_bfloat16, rope_bf16, rope_i_bf16, rope_thd_bf16)
SUM_OP(__nv_bfloat16, sum_bf16)

// Use vectorized fast_sum for bf16, original for other ops
extern "C" __global__ void fast_sum_bf16(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __nv_bfloat16 *src,
    __nv_bfloat16 *dst) {
  fast_sum_bf16_vec(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}
extern "C" __global__ void fast_min_bf16(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __nv_bfloat16 *src,
    __nv_bfloat16 *dst) {
  fast_min(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}
extern "C" __global__ void fast_max_bf16(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __nv_bfloat16 *src,
    __nv_bfloat16 *dst) {
  fast_max(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}
extern "C" __global__ void fast_argmin_bf16(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __nv_bfloat16 *src,
    uint32_t *dst) {
  fast_argmin(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}
extern "C" __global__ void fast_argmax_bf16(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __nv_bfloat16 *src,
    uint32_t *dst) {
  fast_argmax(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}

// NOTE: No reduce ops for f8
// SUM_OP(__nv_fp8_e4m3, sum_fp8_e4m3)
// SOFTMAX_OP(__nv_fp8_e4m3, float, softmax_fp8_e4m3)
// RMSNORM_OP(__nv_fp8_e4m3, rmsnorm_fp8_e4m3)
// LAYERNORM_OP(__nv_fp8_e4m3, layernorm_fp8_e4m3)
// ROPE_OP(__nv_fp8_e4m3, rope_fp8_e4m3, rope_i_fp8_e4m3, rope_thd_fp8_e4m3)
// FAST_OP(__nv_fp8_e4m3, fast_min_fp8_e4m3, fast_max_fp8_e4m3, fast_argmin_fp8_e4m3, fast_argmax_fp8_e4m3, fast_sum_fp8_e4m3)
#endif

#if __CUDA_ARCH__ >= 530
SOFTMAX_OP(__half, float, softmax_f16)
RMSNORM_OP(__half, rmsnorm_f16)
LAYERNORM_OP(__half, layernorm_f16)
ROPE_OP(__half, rope_f16, rope_i_f16, rope_thd_f16)
SUM_OP(__half, sum_f16)
FAST_OP(__half, fast_min_f16, fast_max_f16, fast_argmin_f16, fast_argmax_f16, fast_sum_f16)
#endif

SUM_OP(float, sum_f32)
SUM_OP(double, sum_f64)
SUM_OP(uint32_t, sum_u32)
SOFTMAX_OP(float, float, softmax_f32)
SOFTMAX_OP(double, double, softmax_f64)
RMSNORM_OP(float, rmsnorm_f32)
RMSNORM_OP(double, rmsnorm_f64)
LAYERNORM_OP(float, layernorm_f32)
LAYERNORM_OP(double, layernorm_f64)
ROPE_OP(float, rope_f32, rope_i_f32, rope_thd_f32)
ROPE_OP(double, rope_f64, rope_i_f64, rope_thd_f64)

// Vectorized fast_sum for f32: 4 elements per float4 load
extern "C" __global__ void fast_sum_f32(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const float *src, float *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  float local_sum = 0.0f;
  if (is_contiguous(num_dims, dims, strides)) {
    const size_t vec_start = (start_idx + 3) / 4;
    const size_t vec_stop = stop_idx / 4;

    // Scalar head
    size_t head_end = min(vec_start * 4, stop_idx);
    for (size_t i = start_idx + tid; i < head_end; i += blockDim.x)
      local_sum += src[i];

    if (vec_start < vec_stop && is_aligned_16(src)) {
      const float4 *src4 = reinterpret_cast<const float4*>(src);
      for (size_t vi = vec_start + tid; vi < vec_stop; vi += blockDim.x) {
        float4 v = src4[vi];
        local_sum += v.x + v.y + v.z + v.w;
      }
    } else {
      for (size_t i = head_end + tid; i < vec_stop * 4; i += blockDim.x)
        local_sum += src[i];
    }

    // Scalar tail
    size_t tail_start = vec_stop * 4;
    for (size_t i = tail_start + tid; i < stop_idx; i += blockDim.x)
      local_sum += src[i];
  } else {
    size_t idx = start_idx + tid;
    while (idx < stop_idx) {
      size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
      local_sum += src[strided_i];
      idx += blockDim.x;
    }
  }
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
  __shared__ float shr[32];
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  if (lane_id == 0) shr[warp_id] = local_sum;
  __syncthreads();
  int num_warps = blockDim.x / WARP_SIZE;
  if (tid < WARP_SIZE) {
    local_sum = (tid < num_warps) ? shr[tid] : 0.0f;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
      local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if (tid == 0) dst[dst_id] = local_sum;
  }
}
extern "C" __global__ void fast_min_f32(const size_t src_numel, const size_t el_to_sum_per_block, const size_t num_dims, const size_t *info, const float *src, float *dst) { fast_min(src_numel, el_to_sum_per_block, num_dims, info, src, dst); }
extern "C" __global__ void fast_max_f32(const size_t src_numel, const size_t el_to_sum_per_block, const size_t num_dims, const size_t *info, const float *src, float *dst) { fast_max(src_numel, el_to_sum_per_block, num_dims, info, src, dst); }
extern "C" __global__ void fast_argmin_f32(const size_t src_numel, const size_t el_to_sum_per_block, const size_t num_dims, const size_t *info, const float *src, uint32_t *dst) { fast_argmin(src_numel, el_to_sum_per_block, num_dims, info, src, dst); }
extern "C" __global__ void fast_argmax_f32(const size_t src_numel, const size_t el_to_sum_per_block, const size_t num_dims, const size_t *info, const float *src, uint32_t *dst) { fast_argmax(src_numel, el_to_sum_per_block, num_dims, info, src, dst); }
FAST_OP(double, fast_min_f64, fast_max_f64, fast_argmin_f64, fast_argmax_f64, fast_sum_f64)
FAST_OP(uint32_t, fast_min_u32, fast_max_u32, fast_argmin_u32, fast_argmax_u32, fast_sum_u32)
FAST_OP(int64_t, fast_min_i64, fast_max_i64, fast_argmin_i64, fast_argmax_i64, fast_sum_i64)
FAST_OP(uint8_t, fast_min_u8, fast_max_u8, fast_argmin_u8, fast_argmax_u8, fast_sum_u8)
