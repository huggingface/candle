#include "cuda_utils.cuh"
#include<stdint.h>

// Naive implementation of conv1d.
template <typename T, typename A>
__device__ void conv1d(
    const size_t src_numel,
    const size_t l_out,
    const size_t stride,
    const size_t padding,
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
  // src: (b_size, c_in, l_in)
  // k: (c_out, c_in, k_size)
  const size_t *src_dims = info;
  const size_t *src_s = info + 3;
  const size_t *k_dims = info + 6;
  const size_t *k_s = info + 9;
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t k_size = k_dims[2];
  const size_t c_out = k_dims[0];
  const size_t c_in = src_dims[1];
  const size_t l_in = src_dims[2];
  if (dst_i >= src_dims[0] * c_out * l_out) {
    return;
  }

  // TODO
  const size_t b_idx = dst_i / (l_out * c_out);
  const size_t dst_c_idx = (dst_i / l_out) % c_out;
  const size_t dst_l = dst_i % l_out;

  const size_t src_idx0 = b_idx * src_s[0];
  A d = 0;
  for (size_t offset = 0; offset < k_size; ++offset) {
    size_t src_l = stride * dst_l + offset;
    if (src_l < padding || src_l >= padding + l_in) {
      continue;
    }
    src_l -= padding;
    for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
      const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + src_l * src_s[2];
      const size_t k_idx = dst_c_idx * k_s[0] + src_c_idx * k_s[1] + offset * k_s[2];
      d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
    }
  }
  dst[dst_i] = static_cast<T>(d);
}

// Naive implementation of conv2d.
template <typename T, typename A>
__device__ void conv2d(
    const size_t src_numel,
    const size_t w_out,
    const size_t h_out,
    const size_t stride,
    const size_t padding,
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  // src: (b_size, c_in, w_in, h_in)
  // k: (c_out, c_in, w_k, h_k)
  const size_t *src_dims = info;
  const size_t *src_s = info + 4;
  const size_t *k_dims = info + 8;
  const size_t *k_s = info + 12;
  const size_t w_k = k_dims[2];
  const size_t h_k = k_dims[3];
  const size_t c_out = k_dims[0];
  const size_t c_in = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];
  if (dst_i >= src_dims[0] * c_out * w_out * h_out) {
    return;
  }

  // TODO
  const size_t b_idx = dst_i / (w_out * h_out * c_out);
  const size_t dst_c_idx = (dst_i / (w_out * h_out)) % c_out;
  const size_t dst_w = (dst_i / h_out) % w_out;
  const size_t dst_h = dst_i % h_out;

  const size_t src_idx0 = b_idx * src_s[0];
  A d = 0;
  for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
    size_t src_w = stride * dst_w + w_offset;
    if (src_w < padding || src_w >= w_in + padding) {
      continue;
    }
    src_w -= padding;
    for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
      size_t src_h = stride * dst_h + h_offset;
      if (src_h < padding || src_h >= h_in + padding) {
        continue;
      }
      src_h -= padding;
      for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
        const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + src_w * src_s[2] + src_h * src_s[3];
        const size_t k_idx = dst_c_idx * k_s[0] + src_c_idx * k_s[1] + w_offset * k_s[2] + h_offset * k_s[3];
        d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
      }
    }
  }
  dst[dst_i] = static_cast<T>(d);
}

template <typename T, typename A>
__device__ void avg_pool2d(
    const size_t src_numel,
    const size_t w_k,
    const size_t h_k,
    const size_t w_stride,
    const size_t h_stride,
    const size_t *info,
    const T *src,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  // src: (b_size, c_in, w_in, h_in)
  const size_t *src_dims = info;
  const size_t *src_s = info + 4;

  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  const size_t w_out = (w_in - w_k) / w_stride + 1;
  const size_t h_out = (h_in - h_k) / h_stride + 1;
  if (dst_i >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  // TODO: Improve this.
  const size_t b_idx = dst_i / (w_out * h_out * c);
  const size_t c_idx = (dst_i / (w_out * h_out)) % c;
  const size_t dst_w = (dst_i / h_out) % w_out;
  const size_t dst_h = dst_i % h_out;

  const size_t src_idx0 = b_idx * src_s[0];
  const float scale = 1.0 / (w_k * h_k);
  A d = 0;
  for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
    size_t src_w = w_stride * dst_w + w_offset;
    if (src_w < w_k / 2 || src_w >= w_in + w_k / 2) {
      continue;
    }
    src_w -= w_k / 2;
    for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
      size_t src_h = h_stride * dst_h + h_offset;
      if (src_h < h_k / 2 || src_h >= h_in + h_k / 2) {
        continue;
      }
      src_h -= h_k / 2;
      const size_t src_idx = src_idx0 + c_idx * src_s[1] + src_w * src_s[2] + src_h * src_s[3];
      d += static_cast<A>(src[src_idx]);
    }
  }
  dst[dst_i] = static_cast<T>(d * scale);
}


#define CONV1D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t num_dims, \
    const size_t stride, \
    const size_t padding, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) {  \
  conv1d<TYPENAME, TYPEACC>(src_numel, num_dims, stride, padding, info, src, kernel, dst); \
} \

#define CONV2D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t w_out, \
    const size_t h_out, \
    const size_t stride, \
    const size_t padding, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) {  \
  conv2d<TYPENAME, TYPEACC>(src_numel, w_out, h_out, stride, padding, info, src, kernel, dst); \
} \

#define AVG_POOL2D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t w_k, \
    const size_t h_k, \
    const size_t w_stride, \
    const size_t h_stride, \
    const size_t *info, \
    const TYPENAME *src, \
    TYPENAME *dst \
) {  \
  avg_pool2d<TYPENAME, TYPEACC>(src_numel, w_k, h_k, w_stride, h_stride, info, src, dst); \
} \

#if __CUDA_ARCH__ >= 800
CONV1D_OP(__nv_bfloat16, float, conv1d_bf16)
CONV2D_OP(__nv_bfloat16, float, conv2d_bf16)
AVG_POOL2D_OP(__nv_bfloat16, float, avg_pool2d_bf16)
#endif

#if __CUDA_ARCH__ >= 530
CONV1D_OP(__half, float, conv1d_f16)
CONV2D_OP(__half, float, conv2d_f16)
AVG_POOL2D_OP(__half, float, avg_pool2d_f16)
#endif

CONV1D_OP(float, float, conv1d_f32)
CONV1D_OP(double, double, conv1d_f64)
CONV1D_OP(uint8_t, uint8_t, conv1d_u8)
CONV1D_OP(uint32_t, uint32_t, conv1d_u32)

CONV2D_OP(float, float, conv2d_f32)
CONV2D_OP(double, double, conv2d_f64)
CONV2D_OP(uint8_t, uint8_t, conv2d_u8)
CONV2D_OP(uint32_t, uint32_t, conv2d_u32)

AVG_POOL2D_OP(float, float, avg_pool2d_f32)
AVG_POOL2D_OP(double, double, avg_pool2d_f64)
AVG_POOL2D_OP(uint8_t, uint8_t, avg_pool2d_u8)
AVG_POOL2D_OP(uint32_t, uint32_t, avg_pool2d_u32)
