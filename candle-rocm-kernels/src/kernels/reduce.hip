#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#else
#include <hip/hip_runtime.h>
#endif

#include <stddef.h>
#include <stdint.h>
#include <limits>
#include <math.h>

const int BLOCK_SIZE = 1024;
const int WARP_SIZE = 32;

template <typename T>
__device__ __forceinline__ T reduce_init_lowest() {
    return -INFINITY;
}

template <typename T>
__device__ __forceinline__ T reduce_init_highest() {
    return INFINITY;
}

template <>
__device__ __forceinline__ int64_t reduce_init_lowest<int64_t>() {
    return std::numeric_limits<int64_t>::lowest();
}

template <>
__device__ __forceinline__ uint32_t reduce_init_lowest<uint32_t>() {
    return std::numeric_limits<uint32_t>::lowest();
}

template <>
__device__ __forceinline__ uint8_t reduce_init_lowest<uint8_t>() {
    return std::numeric_limits<uint8_t>::lowest();
}

template <>
__device__ __forceinline__ int64_t reduce_init_highest<int64_t>() {
    return std::numeric_limits<int64_t>::max();
}

template <>
__device__ __forceinline__ uint32_t reduce_init_highest<uint32_t>() {
    return std::numeric_limits<uint32_t>::max();
}

template <>
__device__ __forceinline__ uint8_t reduce_init_highest<uint8_t>() {
    return std::numeric_limits<uint8_t>::max();
}

__device__ bool is_contiguous(
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    size_t acc = 1;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        if (dims[dim_idx] > 1 && acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    }
    return true;
}

__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    unsigned int tmp_i = idx;
    for (int d = num_dims - 1; d >= 0; d--) {
        unsigned int i_dim = tmp_i % dims[d];
        strided_i += i_dim * strides[d];
        tmp_i /= dims[d];
    }
    return strided_i;
}

// Helper functions for reduce operations
template <typename T>
__device__ __forceinline__ T maxg(T a, T b) { return a > b ? a : b; }

template <typename T>
__device__ __forceinline__ T ming(T a, T b) { return a < b ? a : b; }

template <>
__device__ __forceinline__ float maxg<float>(float a, float b) { return fmaxf(a, b); }

template <>
__device__ __forceinline__ double maxg<double>(double a, double b) { return fmax(a, b); }

template <>
__device__ __forceinline__ float ming<float>(float a, float b) { return fminf(a, b); }

template <>
__device__ __forceinline__ double ming<double>(double a, double b) { return fmin(a, b); }

__device__ __forceinline__ float expg(float x) { return expf(x); }
__device__ __forceinline__ double expg(double x) { return exp(x); }

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, 32);
    }
    return val;
}

__device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        a.x += __shfl_xor(a.x, offset, 32);
        a.y += __shfl_xor(a.y, offset, 32);
    }
    return a;
}

template <typename T>
__device__ void
fast_sum(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, T *dst) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;

    __shared__ T shr[BLOCK_SIZE];
    size_t tid = threadIdx.x;
    size_t dst_id = blockIdx.x;

    shr[tid] = 0;
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    while (idx < stop_idx) {
        size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
        shr[tid] += src[strided_i];
        idx += blockDim.x;
    }

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s)
            shr[tid] += shr[tid + s];
    }

    if (tid == 0)
        dst[dst_id] = shr[0];
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

    shr[tid] = reduce_init_lowest<T>();
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    while (idx < stop_idx) {
        size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
        shr[tid] = maxg(shr[tid], src[strided_i]);
        idx += blockDim.x;
    }

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

    shr[tid] = reduce_init_highest<T>();
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    while (idx < stop_idx) {
        size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
        shr[tid] = ming(shr[tid], src[strided_i]);
        idx += blockDim.x;
    }

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

    shr[tid] = reduce_init_highest<T>();
    shr_index[tid] = 0xFFFFFFFF;
    bool not_set = true;
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    while (idx < stop_idx) {
        size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
        if (not_set || src[strided_i] < shr[tid]) {
            shr[tid] = src[strided_i];
            shr_index[tid] = idx % dims[num_dims - 1];
            not_set = false;
        }
        idx += blockDim.x;
    }

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

    shr[tid] = reduce_init_lowest<T>();
    shr_index[tid] = 0xFFFFFFFF;
    bool not_set = true;
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    while (idx < stop_idx) {
        size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
        if (not_set || src[strided_i] > shr[tid]) {
            shr[tid] = src[strided_i];
            shr_index[tid] = idx % dims[num_dims - 1];
            not_set = false;
        }
        idx += blockDim.x;
    }

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

// Softmax implementation adapted from ggml.
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
    for (int mask = 16; mask > 0; mask >>= 1) {
        max_val = maxg(max_val, __shfl_xor(max_val, mask, 32));
    }

    ACC tmp = 0.;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        const T val = expg(x[i] - max_val);
        tmp += (ACC)val;
        dst[i] = val;
    }

    // sum up partial sums
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor(tmp, mask, 32);
    }

    const ACC inv_tmp = 1. / tmp;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        dst[i] *= inv_tmp;
    }
}

// RmsNorm implementation adapted from ggml, accumulation is made using f32.
template <typename T>
__device__ void rmsnorm(const T * x, T * dst, const T * alpha, const int ncols, const int block_size, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = (float)(x[row*ncols + col]);
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
          dst[row*ncols + col] = (T)(scale * (float)(x[row*ncols + col]));
      }
    }
    else {
      for (int col = tid; col < ncols; col += block_size) {
          float a = (float)(alpha[col]);
          dst[row*ncols + col] = (T)(scale * (float)(x[row*ncols + col]) * a);
      }
    }
}

// TODO: Replace with MIOpen implementation once rocm-rs exposes miopenLayerNorm
template <typename T>
__device__ void layernorm(const T * x, T * dst, const T * alpha, const T * beta,
                          const int ncols, const int block_size, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float2 mean_var = make_float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = (float)(x[row*ncols + col]);
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

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
          float lhs = ((float)(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = (T)(lhs);
      }
    }
    else if (alpha == nullptr && beta != nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float b = (float)(beta[col]);
          float lhs = ((float)(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = (T)(lhs + b);
      }
    }
    else if (alpha != nullptr && beta == nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float a = (float)(alpha[col]);
          float lhs = ((float)(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = (T)(lhs * a);
      }
    }
    else {
      for (int col = tid; col < ncols; col += block_size) {
          float a = (float)(alpha[col]);
          float b = (float)(beta[col]);
          float lhs = ((float)(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = (T)(lhs * a + b);
      }
    }
}

#define FAST_OP(TYPENAME, MIN_NAME, MAX_NAME, ARGMIN_NAME, ARGMAX_NAME, SUM_NAME) \
  extern "C" __global__ void ARGMIN_NAME( \
      const size_t src_numel, const size_t el_to_sum_per_block, \
      const size_t num_dims, const size_t *info, const TYPENAME *src, \
      uint32_t *dst) { \
    fast_argmin(src_numel, el_to_sum_per_block, num_dims, info, src, dst); \
  } \
  extern "C" __global__ void ARGMAX_NAME( \
      const size_t src_numel, const size_t el_to_sum_per_block, \
      const size_t num_dims, const size_t *info, const TYPENAME *src, \
      uint32_t *dst) { \
    fast_argmax(src_numel, el_to_sum_per_block, num_dims, info, src, dst); \
  } \
  extern "C" __global__ void MIN_NAME( \
      const size_t src_numel, const size_t el_to_sum_per_block, \
      const size_t num_dims, const size_t *info, const TYPENAME *src, \
      TYPENAME *dst) { \
    fast_min(src_numel, el_to_sum_per_block, num_dims, info, src, dst); \
  } \
  extern "C" __global__ void MAX_NAME( \
      const size_t src_numel, const size_t el_to_sum_per_block, \
      const size_t num_dims, const size_t *info, const TYPENAME *src, \
      TYPENAME *dst) { \
    fast_max(src_numel, el_to_sum_per_block, num_dims, info, src, dst); \
  } \
  extern "C" __global__ void SUM_NAME( \
      const size_t src_numel, const size_t el_to_sum_per_block, \
      const size_t num_dims, const size_t *info, const TYPENAME *src, \
      TYPENAME *dst) { \
    fast_sum(src_numel, el_to_sum_per_block, num_dims, info, src, dst); \
  }

#define SOFTMAX_OP(TYPENAME, ACC_TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME( \
      const TYPENAME *src, TYPENAME *dst, \
      const int n_cols) { \
    softmax<TYPENAME, ACC_TYPENAME>(src, dst, n_cols); \
  }

#define RMSNORM_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME( \
      const TYPENAME *src, TYPENAME *dst, const TYPENAME *alpha, \
      const int n_cols, const int block_size, const float eps) { \
    rmsnorm<TYPENAME>(src, dst, alpha, n_cols, block_size, eps); \
  }

#define LAYERNORM_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME( \
      const TYPENAME *src, TYPENAME *dst, const TYPENAME *alpha, \
      const TYPENAME *beta, const int n_cols, const int block_size, const float eps) { \
    layernorm<TYPENAME>(src, dst, alpha, beta, n_cols, block_size, eps); \
  }

FAST_OP(float, fast_min_f32, fast_max_f32, fast_argmin_f32, fast_argmax_f32, fast_sum_f32)
FAST_OP(double, fast_min_f64, fast_max_f64, fast_argmin_f64, fast_argmax_f64, fast_sum_f64)
FAST_OP(uint32_t, fast_min_u32, fast_max_u32, fast_argmin_u32, fast_argmax_u32, fast_sum_u32)
FAST_OP(int64_t, fast_min_i64, fast_max_i64, fast_argmin_i64, fast_argmax_i64, fast_sum_i64)
FAST_OP(uint8_t, fast_min_u8, fast_max_u8, fast_argmin_u8, fast_argmax_u8, fast_sum_u8)

// Softmax kernels
SOFTMAX_OP(float, float, softmax_f32)
SOFTMAX_OP(double, double, softmax_f64)

// RmsNorm kernels
RMSNORM_OP(float, rmsnorm_f32)
RMSNORM_OP(double, rmsnorm_f64)

// LayerNorm kernels
LAYERNORM_OP(float, layernorm_f32)
LAYERNORM_OP(double, layernorm_f64)

// Half-precision types support (HIP uses __half for f16 and hip_bfloat16 for bf16)
#ifdef __HIPCC__
#include <hip/hip_fp16.h>

// F16 helpers
__device__ __forceinline__ float maxg(__half a, __half b) {
    return fmaxf(__half2float(a), __half2float(b));
}

__device__ __forceinline__ float maxg(float a, __half b) {
    return fmaxf(a, __half2float(b));
}

__device__ __forceinline__ float expg(__half x) { return expf(__half2float(x)); }

// Softmax for F16 (accumulate in float)
template <>
__device__ void softmax<__half, float>(const __half * x, __half * dst, const int ncols) {
    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;

    float max_val = -INFINITY;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        max_val = fmaxf(max_val, __half2float(x[i]));
    }

    // find the max value in the block
    for (int mask = 16; mask > 0; mask >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor(max_val, mask, 32));
    }

    float tmp = 0.;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        const float val = expf(__half2float(x[i]) - max_val);
        tmp += val;
        dst[i] = __float2half(val);
    }

    // sum up partial sums
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor(tmp, mask, 32);
    }

    const float inv_tmp = 1. / tmp;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        dst[i] = __float2half(__half2float(dst[i]) * inv_tmp);
    }
}

// RmsNorm for F16
template <>
__device__ void rmsnorm<__half>(const __half * x, __half * dst, const __half * alpha, const int ncols, const int block_size, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f;

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = __half2float(x[row*ncols + col]);
        tmp += xi * xi;
    }

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
          dst[row*ncols + col] = __float2half(scale * __half2float(x[row*ncols + col]));
      }
    }
    else {
      for (int col = tid; col < ncols; col += block_size) {
          float a = __half2float(alpha[col]);
          dst[row*ncols + col] = __float2half(scale * __half2float(x[row*ncols + col]) * a);
      }
    }
}

// Kernel wrappers for half-precision (F16 only - BF16 requires newer ROCm)
extern "C" __global__ void softmax_f16(const __half *src, __half *dst, const int n_cols) {
    softmax<__half, float>(src, dst, n_cols);
}

extern "C" __global__ void rmsnorm_f16(const __half *src, __half *dst, const __half *alpha, const int n_cols, const int block_size, const float eps) {
    rmsnorm<__half>(src, dst, alpha, n_cols, block_size, eps);
}

// LayerNorm for F16
template <>
__device__ void layernorm<__half>(const __half * x, __half * dst, const __half * alpha, const __half * beta,
                                  const int ncols, const int block_size, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float2 mean_var = make_float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = __half2float(x[row*ncols + col]);
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

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
          float lhs = (__half2float(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = __float2half(lhs);
      }
    }
    else if (alpha == nullptr && beta != nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float b = __half2float(beta[col]);
          float lhs = (__half2float(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = __float2half(lhs + b);
      }
    }
    else if (alpha != nullptr && beta == nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float a = __half2float(alpha[col]);
          float lhs = (__half2float(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = __float2half(lhs * a);
      }
    }
    else {
      for (int col = tid; col < ncols; col += block_size) {
          float a = __half2float(alpha[col]);
          float b = __half2float(beta[col]);
          float lhs = (__half2float(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = __float2half(lhs * a + b);
      }
    }
}

extern "C" __global__ void layernorm_f16(const __half *src, __half *dst, const __half *alpha,
                                         const __half *beta, const int n_cols, const int block_size, const float eps) {
    layernorm<__half>(src, dst, alpha, beta, n_cols, block_size, eps);
}

#endif // __HIPCC__
