// Pooling and upsampling over NCHW tensors. The source layout (offset + 4
// strides) is passed explicitly so views work without a contiguous copy.
#include "common.hpp"

namespace {
struct NCHW {
  int64_t off, s0, s1, s2, s3;
  int64_t b, c, h, w;
};
} // namespace

extern "C" {

int candle_sycl_avg_pool2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp, void *out,
                           const int64_t *src, size_t k_h, size_t k_w, size_t s_h, size_t s_w,
                           size_t h_out, size_t w_out) {
  NCHW L{src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8]};
  size_t numel = (size_t)L.b * L.c * h_out * w_out;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    using A = typename Acc<T>::type;
    A scale = A(1) / (A)(k_h * k_w);
    const T *in = static_cast<const T *>(inp);
    T *o = static_cast<T *>(out);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        size_t i = gid[0];
        size_t wo = i % w_out;
        size_t ho = (i / w_out) % h_out;
        size_t cc = (i / (w_out * h_out)) % L.c;
        size_t bb = i / (w_out * h_out * L.c);
        int64_t base = L.off + (int64_t)bb * L.s0 + (int64_t)cc * L.s1;
        A sum = 0;
        for (size_t m = 0; m < k_h; ++m)
          for (size_t n = 0; n < k_w; ++n)
            sum += (A)in[base + (int64_t)(s_h * ho + m) * L.s2 +
                         (int64_t)(s_w * wo + n) * L.s3];
        o[i] = static_cast<T>(sum * scale);
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

int candle_sycl_max_pool2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp, void *out,
                           const int64_t *src, size_t k_h, size_t k_w, size_t s_h, size_t s_w,
                           size_t h_out, size_t w_out) {
  NCHW L{src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8]};
  size_t numel = (size_t)L.b * L.c * h_out * w_out;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    T *o = static_cast<T *>(out);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        size_t i = gid[0];
        size_t wo = i % w_out;
        size_t ho = (i / w_out) % h_out;
        size_t cc = (i / (w_out * h_out)) % L.c;
        size_t bb = i / (w_out * h_out * L.c);
        int64_t base = L.off + (int64_t)bb * L.s0 + (int64_t)cc * L.s1;
        T best = in[base + (int64_t)(s_h * ho) * L.s2 + (int64_t)(s_w * wo) * L.s3];
        for (size_t m = 0; m < k_h; ++m)
          for (size_t n = 0; n < k_w; ++n) {
            T v = in[base + (int64_t)(s_h * ho + m) * L.s2 + (int64_t)(s_w * wo + n) * L.s3];
            if (best < v) best = v;
          }
        o[i] = best;
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

int candle_sycl_upsample_nearest2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                                   void *out, const int64_t *src, size_t dst_h, size_t dst_w) {
  NCHW L{src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8]};
  size_t numel = (size_t)L.b * L.c * dst_h * dst_w;
  double sh = (double)L.h / (double)dst_h, sw = (double)L.w / (double)dst_w;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    T *o = static_cast<T *>(out);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        size_t i = gid[0];
        size_t wo = i % dst_w;
        size_t ho = (i / dst_w) % dst_h;
        size_t cc = (i / (dst_w * dst_h)) % L.c;
        size_t bb = i / (dst_w * dst_h * L.c);
        size_t sy = sycl::min((size_t)(L.h - 1), (size_t)((double)ho * sh));
        size_t sx = sycl::min((size_t)(L.w - 1), (size_t)((double)wo * sw));
        o[i] = in[L.off + (int64_t)bb * L.s0 + (int64_t)cc * L.s1 + (int64_t)sy * L.s2 +
                  (int64_t)sx * L.s3];
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

// src is (b, c, w) here: s2 is the width stride, h/s2... reuse NCHW with h=1.
int candle_sycl_upsample_nearest1d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                                   void *out, const int64_t *src, size_t dst_w) {
  int64_t off = src[0], s0 = src[1], s1 = src[2], sw = src[3];
  int64_t b = src[4], c = src[5], w_in = src[6];
  size_t numel = (size_t)b * c * dst_w;
  double sc = (double)w_in / (double)dst_w;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    T *o = static_cast<T *>(out);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        size_t i = gid[0];
        size_t wo = i % dst_w;
        size_t cc = (i / dst_w) % c;
        size_t bb = i / (dst_w * c);
        size_t sx = sycl::min((size_t)(w_in - 1), (size_t)((double)wo * sc));
        o[i] = in[off + (int64_t)bb * s0 + (int64_t)cc * s1 + (int64_t)sx * sw];
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

int candle_sycl_upsample_bilinear2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                                    void *out, const int64_t *src, size_t dst_h, size_t dst_w,
                                    int align_corners, double scale_h, double scale_w) {
  NCHW L{src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8]};
  size_t numel = (size_t)L.b * L.c * dst_h * dst_w;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    T *o = static_cast<T *>(out);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        size_t i = gid[0];
        size_t wo = i % dst_w;
        size_t ho = (i / dst_w) % dst_h;
        size_t cc = (i / (dst_w * dst_h)) % L.c;
        size_t bb = i / (dst_w * dst_h * L.c);
        auto coord = [&](size_t out_i, double scale) {
          return align_corners ? scale * (double)out_i
                               : scale * ((double)out_i + 0.5) - 0.5;
        };
        double sy = sycl::fmax(coord(ho, scale_h), 0.0);
        double sx = sycl::fmax(coord(wo, scale_w), 0.0);
        size_t y0 = (size_t)sy, x0 = (size_t)sx;
        size_t y1 = sycl::min(y0 + 1, (size_t)(L.h - 1));
        size_t x1 = sycl::min(x0 + 1, (size_t)(L.w - 1));
        double wy = sycl::clamp(sy - (double)y0, 0.0, 1.0);
        double wx = sycl::clamp(sx - (double)x0, 0.0, 1.0);
        int64_t base = L.off + (int64_t)bb * L.s0 + (int64_t)cc * L.s1;
        double v00 = (double)in[base + (int64_t)y0 * L.s2 + (int64_t)x0 * L.s3];
        double v10 = (double)in[base + (int64_t)y0 * L.s2 + (int64_t)x1 * L.s3];
        double v01 = (double)in[base + (int64_t)y1 * L.s2 + (int64_t)x0 * L.s3];
        double v11 = (double)in[base + (int64_t)y1 * L.s2 + (int64_t)x1 * L.s3];
        double top = v00 * (1.0 - wx) + v10 * wx;
        double bot = v01 * (1.0 - wx) + v11 * wx;
        o[i] = static_cast<T>(top * (1.0 - wy) + bot * wy);
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

} // extern "C"
