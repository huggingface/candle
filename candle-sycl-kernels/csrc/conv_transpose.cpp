// Direct (output-centric) transposed convolution. Correctness path; a col2im +
// GEMM formulation or oneDNN would be faster.
#include "common.hpp"

namespace {
struct T2D {
  int64_t off, s0, s1, s2, s3;
};
struct T1D {
  int64_t off, s0, s1, s2;
};
} // namespace

extern "C" {

// inp (b, c_in, ih, iw) strided; kernel (c_in, c_out, kh, kw) contiguous.
// out (b, c_out, out_h, out_w) dense.
int candle_sycl_conv_transpose2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                                 const int64_t *im, const void *ker, void *out, size_t b,
                                 size_t c_in, size_t c_out, size_t ih, size_t iw, size_t kh,
                                 size_t kw, size_t out_h, size_t out_w, size_t stride,
                                 size_t padding, size_t dilation) {
  T2D L{im[0], im[1], im[2], im[3], im[4]};
  size_t numel = b * c_out * out_h * out_w;
  int64_t ks0 = (int64_t)c_out * kh * kw, ks1 = (int64_t)kh * kw, ks2 = (int64_t)kw;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    const T *k = static_cast<const T *>(ker);
    T *o = static_cast<T *>(out);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        size_t idx = gid[0];
        size_t ox = idx % out_w;
        size_t oy = (idx / out_w) % out_h;
        size_t co = (idx / (out_w * out_h)) % c_out;
        size_t bb = idx / (out_w * out_h * c_out);
        double acc = 0;
        for (size_t ky = 0; ky < kh; ++ky) {
          long num_y = (long)(oy + padding) - (long)(ky * dilation);
          if (num_y < 0 || (size_t)num_y % stride) continue;
          size_t iy = (size_t)num_y / stride;
          if (iy >= ih) continue;
          for (size_t kx = 0; kx < kw; ++kx) {
            long num_x = (long)(ox + padding) - (long)(kx * dilation);
            if (num_x < 0 || (size_t)num_x % stride) continue;
            size_t ix = (size_t)num_x / stride;
            if (ix >= iw) continue;
            for (size_t ci = 0; ci < c_in; ++ci) {
              double v = (double)in[L.off + (int64_t)bb * L.s0 + (int64_t)ci * L.s1 +
                                    (int64_t)iy * L.s2 + (int64_t)ix * L.s3];
              double w = (double)k[(int64_t)ci * ks0 + (int64_t)co * ks1 +
                                   (int64_t)ky * ks2 + (int64_t)kx];
              acc += v * w;
            }
          }
        }
        o[idx] = static_cast<T>(acc);
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

int candle_sycl_conv_transpose1d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                                 const int64_t *im, const void *ker, void *out, size_t b,
                                 size_t c_in, size_t c_out, size_t il, size_t kl, size_t out_l,
                                 size_t stride, size_t padding, size_t dilation) {
  T1D L{im[0], im[1], im[2], im[3]};
  size_t numel = b * c_out * out_l;
  int64_t ks0 = (int64_t)c_out * kl, ks1 = (int64_t)kl;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    const T *k = static_cast<const T *>(ker);
    T *o = static_cast<T *>(out);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        size_t idx = gid[0];
        size_t ol = idx % out_l;
        size_t co = (idx / out_l) % c_out;
        size_t bb = idx / (out_l * c_out);
        double acc = 0;
        for (size_t kk = 0; kk < kl; ++kk) {
          long num = (long)(ol + padding) - (long)(kk * dilation);
          if (num < 0 || (size_t)num % stride) continue;
          size_t ii = (size_t)num / stride;
          if (ii >= il) continue;
          for (size_t ci = 0; ci < c_in; ++ci) {
            double v = (double)in[L.off + (int64_t)bb * L.s0 + (int64_t)ci * L.s1 +
                                  (int64_t)ii * L.s2];
            double w = (double)k[(int64_t)ci * ks0 + (int64_t)co * ks1 + (int64_t)kk];
            acc += v * w;
          }
        }
        o[idx] = static_cast<T>(acc);
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

} // extern "C"
