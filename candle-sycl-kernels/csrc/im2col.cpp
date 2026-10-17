// im2col for the conv path (conv = im2col + GEMM + permute, same strategy as the
// candle CPU/CUDA backends). oneDNN would be the faster path for large images.
#include "common.hpp"

namespace {
struct S2D {
  int64_t off, s0, s1, s2, s3; // input NCHW strides
  int64_t b, c_in, h, w;
};
}

extern "C" {

// col: (b * out_h * out_w, c_in * k_h * k_w), row-major, zero-padded.
int candle_sycl_im2col2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp, void *col,
                         const int64_t *meta, size_t k_h, size_t k_w, size_t stride,
                         size_t padding, size_t dilation, size_t out_h, size_t out_w) {
  S2D L{meta[0], meta[1], meta[2], meta[3], meta[4], meta[5], meta[6], meta[7], meta[8]};
  size_t kk = L.c_in * k_h * k_w;
  size_t rows = (size_t)L.b * out_h * out_w;
  size_t numel = rows * kk;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    T *o = static_cast<T *>(col);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        size_t idx = gid[0];
        size_t kw = idx % k_w;
        size_t kh = (idx / k_w) % k_h;
        size_t ci = (idx / (k_w * k_h)) % L.c_in;
        size_t row = idx / kk;
        size_t ow = row % out_w;
        size_t oh = (row / out_w) % out_h;
        size_t bb = row / (out_w * out_h);
        long ih = (long)(oh * stride + kh * dilation) - (long)padding;
        long iw = (long)(ow * stride + kw * dilation) - (long)padding;
        T v = T(0);
        if (ih >= 0 && iw >= 0 && ih < L.h && iw < L.w) {
          v = in[L.off + (int64_t)bb * L.s0 + (int64_t)ci * L.s1 + ih * L.s2 + iw * L.s3];
        }
        o[idx] = v;
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

// col: (b * out_l, c_in * k). meta = [off, s0, s1, s2, b, c_in, l].
int candle_sycl_im2col1d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp, void *col,
                         const int64_t *meta, size_t k, size_t stride, size_t padding,
                         size_t dilation, size_t out_l) {
  int64_t off = meta[0], s0 = meta[1], s1 = meta[2], s2 = meta[3];
  int64_t b = meta[4], c_in = meta[5], l = meta[6];
  size_t kk = (size_t)c_in * k;
  size_t rows = (size_t)b * out_l;
  size_t numel = rows * kk;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    T *o = static_cast<T *>(col);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        size_t idx = gid[0];
        size_t kk_i = idx % kk;
        size_t ki = kk_i % k;
        size_t ci = kk_i / k;
        size_t row = idx / kk;
        size_t ol = row % out_l;
        size_t bb = row / out_l;
        long il = (long)(ol * stride + ki * dilation) - (long)padding;
        T v = T(0);
        if (il >= 0 && il < l) {
          v = in[off + (int64_t)bb * s0 + (int64_t)ci * s1 + il * s2];
        }
        o[idx] = v;
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

} // extern "C"
