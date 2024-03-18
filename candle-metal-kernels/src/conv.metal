#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))

template <typename T>
METAL_FUNC void im2col(
    constant size_t &dst_numel,
    constant size_t &h_out,
    constant size_t &w_out,
    constant size_t &h_k,
    constant size_t &w_k,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  // dst: (b_size, h_out, w_out, c_in, h_k, w_k)
  // src: (b_size, c_in, h_in, w_in)
  if (tid >= dst_numel) {
    return;
  }
  const size_t b_in = src_dims[0];
  const size_t c_in = src_dims[1];
  const size_t h_in = src_dims[2];
  const size_t w_in = src_dims[3];

  const size_t dst_s4 = w_k;
  const size_t dst_s3 = h_k * dst_s4;
  const size_t dst_s2 = c_in * dst_s3;
  const size_t dst_s1 = w_out * dst_s2;
  const size_t dst_s0 = h_out * dst_s1;

  size_t tmp_tid = tid;
  const size_t b_idx = tmp_tid / dst_s0;
  tmp_tid -= b_idx * dst_s0;
  const size_t h_idx = tmp_tid / dst_s1;
  tmp_tid -= h_idx * dst_s1;
  const size_t w_idx = tmp_tid / dst_s2;
  tmp_tid -= w_idx * dst_s2;
  const size_t c_idx = tmp_tid / dst_s3;
  tmp_tid -= c_idx * dst_s3;
  const size_t h_k_idx = tmp_tid / dst_s4;
  tmp_tid -= h_k_idx * dst_s4;
  const size_t w_k_idx = tmp_tid;
  size_t src_h_idx = h_idx * stride + h_k_idx * dilation;
  size_t src_w_idx = w_idx * stride + w_k_idx * dilation;
  if (src_h_idx < padding || src_h_idx >= h_in + padding) {
    dst[tid] = static_cast<T>(0);
  }
  else if (src_w_idx < padding || src_w_idx >= w_in + padding) {
    dst[tid] = static_cast<T>(0);
  }
  else {
    src_h_idx -= padding;
    src_w_idx -= padding;
    const size_t src_i =
      b_idx * src_strides[0]
      + c_idx * src_strides[1]
      + src_h_idx * src_strides[2]
      + src_w_idx * src_strides[3];
    dst[tid] = src[src_i];
  }
}

template <typename T>
METAL_FUNC void im2col1d(
    constant size_t &dst_numel,
    constant size_t &l_out,
    constant size_t &l_k,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  // dst: (b_size, l_out, c_in, l_k)
  // src: (b_size, c_in, l_in)
  if (tid >= dst_numel) {
    return;
  }
  const size_t b_in = src_dims[0];
  const size_t c_in = src_dims[1];
  const size_t l_in = src_dims[2];

  const size_t dst_s2 = l_k;
  const size_t dst_s1 = c_in * dst_s2;
  const size_t dst_s0 = l_out * dst_s1;

  size_t tmp_dst_i = tid;
  const size_t b_idx = tmp_dst_i / dst_s0;
  tmp_dst_i -= b_idx * dst_s0;
  const size_t l_idx = tmp_dst_i / dst_s1;
  tmp_dst_i -= l_idx * dst_s1;
  const size_t c_idx = tmp_dst_i / dst_s2;
  tmp_dst_i -= c_idx * dst_s2;
  const size_t l_k_idx = tmp_dst_i;
  size_t src_l_idx = l_idx * stride + l_k_idx * dilation;
  if (src_l_idx < padding || src_l_idx >= l_in + padding) {
    dst[tid] = static_cast<T>(0);
  }
  else {
    src_l_idx -= padding;
    const size_t src_i = b_idx * src_strides[0] + c_idx * src_strides[1] + src_l_idx * src_strides[2];
    dst[tid] = src[src_i];
  }
}

template <typename T>
METAL_FUNC void upsample_nearest2d(
    constant size_t &w_out,
    constant size_t &h_out,
    constant float &w_scale,
    constant float &h_scale,
    constant size_t *src_dims,
    constant size_t *src_s,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  // src: (b_size, c_in, w_in, h_in)

  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  if (tid >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  // TODO: Improve this.
  const size_t b_idx = tid / (w_out * h_out * c);
  const size_t c_idx = (tid / (w_out * h_out)) % c;
  const size_t dst_w = (tid / h_out) % w_out;
  const size_t dst_h = tid % h_out;

  size_t src_w = static_cast<size_t>(dst_w * w_scale);
  size_t src_h = static_cast<size_t>(dst_h * h_scale);
  if (src_w >= w_in) {
    src_w = w_in - 1;
  }
  if (src_h >= h_in) {
    src_h = h_in - 1;
  }

  const size_t src_i = b_idx * src_s[0] + c_idx * src_s[1] + src_w * src_s[2] + src_h * src_s[3];
  dst[tid] = src[src_i];
}

#define IM2COL_OP(T, FN_NAME) \
kernel void FN_NAME(  \
    constant size_t &dst_numel, \
    constant size_t &h_out, \
    constant size_t &w_out, \
    constant size_t &h_k, \
    constant size_t &w_k, \
    constant size_t &stride, \
    constant size_t &padding, \
    constant size_t &dilation, \
    constant size_t *src_dims, \
    constant size_t *src_strides, \
    device const T *src, \
    device T *dst, \
    uint tid [[ thread_position_in_grid ]] \
) {  \
  im2col<T>(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, src_dims, src_strides, src, dst, tid); \
} \

#define IM2COL1D_OP(T, FN_NAME) \
kernel void FN_NAME(  \
    constant size_t &dst_numel, \
    constant size_t &l_out, \
    constant size_t &l_k, \
    constant size_t &stride, \
    constant size_t &padding, \
    constant size_t &dilation, \
    constant size_t *src_dims, \
    constant size_t *src_strides, \
    device const T *src, \
    device T *dst, \
    uint tid [[ thread_position_in_grid ]] \
) {  \
  im2col1d<T>(dst_numel, l_out, l_k, stride, padding, dilation, src_dims, src_strides, src, dst, tid); \
} \
 
#define UPSAMPLE_NEAREST2D_OP(TYPENAME, FN_NAME) \
kernel void FN_NAME(  \
    constant size_t &w_out, \
    constant size_t &h_out, \
    constant float &w_scale, \
    constant float &h_scale, \
    constant size_t *dims, \
    constant size_t *strides, \
    device const TYPENAME *src, \
    device TYPENAME *dst, \
    uint tid [[ thread_position_in_grid ]] \
) {  \
  upsample_nearest2d<TYPENAME>(w_out, h_out, w_scale, h_scale, dims, strides, src, dst, tid); \
} \

template <typename T, typename A>
METAL_FUNC void avg_pool2d(
    constant size_t &w_k,
    constant size_t &h_k,
    constant size_t &w_stride,
    constant size_t &h_stride,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  const size_t w_out = (w_in - w_k) / w_stride + 1;
  const size_t h_out = (h_in - h_k) / h_stride + 1;
  if (tid >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  const size_t b_idx = tid / (w_out * h_out * c);
  const size_t c_idx = (tid / (w_out * h_out)) % c;
  const size_t dst_w = (tid / h_out) % w_out;
  const size_t dst_h = tid % h_out;

  const size_t src_idx0 = b_idx * src_strides[0];
  A d = 0;
  for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
    size_t src_w = w_stride * dst_w + w_offset;
    if (src_w >= w_in){
      continue;
    }
    for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
      size_t src_h = h_stride * dst_h + h_offset;
      if (src_h >= h_in) {
        continue;
      }
      const size_t src_idx = src_idx0 + c_idx * src_strides[1] + src_w * src_strides[2] + src_h * src_strides[3];
      d += static_cast<A>(src[src_idx]);
    }
  }
  dst[tid] = static_cast<T>(d / (w_k * h_k));
}

#define AVGPOOL2D_OP(TYPENAME, TYPEACC, FN_NAME) \
kernel void FN_NAME( \
    constant size_t &w_k, \
    constant size_t &h_k, \
    constant size_t &w_s, \
    constant size_t &h_s, \
    constant size_t *src_dims, \
    constant size_t *src_s, \
    device const TYPENAME *src, \
    device TYPENAME *dst, \
    uint tid [[ thread_position_in_grid ]] \
) { \
  avg_pool2d<TYPENAME, TYPEACC>(w_k, h_k, w_s, h_s, src_dims, src_s, src, dst, tid); \
} \

template <typename T>
METAL_FUNC void max_pool2d(
    constant size_t &w_k,
    constant size_t &h_k,
    constant size_t &w_stride,
    constant size_t &h_stride,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  const size_t w_out = (w_in - w_k) / w_stride + 1;
  const size_t h_out = (h_in - h_k) / h_stride + 1;
  if (tid >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  const size_t b_idx = tid / (w_out * h_out * c);
  const size_t c_idx = (tid / (w_out * h_out)) % c;
  const size_t dst_w = (tid / h_out) % w_out;
  const size_t dst_h = tid % h_out;

  const size_t src_idx0 = b_idx * src_strides[0];
  T d = 0;
  bool set = false;
  for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
    size_t src_w = w_stride * dst_w + w_offset;
    if (src_w >= w_in){
      continue;
    }
    for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
      size_t src_h = h_stride * dst_h + h_offset;
      if (src_h >= h_in) {
        continue;
      }
      const size_t src_idx = src_idx0 + c_idx * src_strides[1] + src_w * src_strides[2] + src_h * src_strides[3];
      if (set) {
        d = MAX(d, src[src_idx]);
      }
      else {
        d = src[src_idx];
        set = true;
      }
    }
  }
  dst[tid] = d;
}

#define MAXPOOL2D_OP(TYPENAME, FN_NAME) \
kernel void FN_NAME( \
    constant size_t &w_k, \
    constant size_t &h_k, \
    constant size_t &w_s, \
    constant size_t &h_s, \
    constant size_t *src_dims, \
    constant size_t *src_s, \
    device const TYPENAME *src, \
    device TYPENAME *dst, \
    uint tid [[ thread_position_in_grid ]] \
) { \
  max_pool2d<TYPENAME>(w_k, h_k, w_s, h_s, src_dims, src_s, src, dst, tid); \
} \

IM2COL_OP(float, im2col_f32)
IM2COL_OP(uint8_t, im2col_u8)
IM2COL_OP(uint32_t, im2col_u32)

IM2COL1D_OP(float, im2col1d_f32)
IM2COL1D_OP(uint8_t, im2col1d_u8)
IM2COL1D_OP(uint32_t, im2col1d_u32)

UPSAMPLE_NEAREST2D_OP(float, upsample_nearest2d_f32)
UPSAMPLE_NEAREST2D_OP(uint8_t, upsample_nearest2d_u8)
UPSAMPLE_NEAREST2D_OP(uint32_t, upsample_nearest2d_u32)

MAXPOOL2D_OP(float, max_pool2d_f32)
MAXPOOL2D_OP(half, max_pool2d_f16)
MAXPOOL2D_OP(uint32_t, max_pool2d_u32)
MAXPOOL2D_OP(uint8_t, max_pool2d_u8)
#if defined(__HAVE_BFLOAT__)
MAXPOOL2D_OP(bfloat, max_pool2d_bf16)
#endif

AVGPOOL2D_OP(float, float, avg_pool2d_f32)
AVGPOOL2D_OP(half, float, avg_pool2d_f16)
AVGPOOL2D_OP(uint32_t, uint32_t, avg_pool2d_u32)
AVGPOOL2D_OP(uint8_t, uint8_t, avg_pool2d_u8)
#if defined(__HAVE_BFLOAT__)
AVGPOOL2D_OP(bfloat, float, avg_pool2d_bf16)
#endif