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

FAST_OP(float, fast_min_f32, fast_max_f32, fast_argmin_f32, fast_argmax_f32, fast_sum_f32)
FAST_OP(double, fast_min_f64, fast_max_f64, fast_argmin_f64, fast_argmax_f64, fast_sum_f64)
FAST_OP(uint32_t, fast_min_u32, fast_max_u32, fast_argmin_u32, fast_argmax_u32, fast_sum_u32)
FAST_OP(int64_t, fast_min_i64, fast_max_i64, fast_argmin_i64, fast_argmax_i64, fast_sum_i64)
FAST_OP(uint8_t, fast_min_u8, fast_max_u8, fast_argmin_u8, fast_argmax_u8, fast_sum_u8)
