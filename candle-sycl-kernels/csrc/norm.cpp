// Fused candle-nn ops: softmax(last dim), rms_norm, rope (3 layout variants).
// Row/pair parallel, all inputs contiguous. Replaces the host round-trip shim.
#include "common.hpp"
#include <cmath>

namespace {
template <typename T> struct FAcc { using type = float; };
template <> struct FAcc<double> { using type = double; };
} // namespace

extern "C" {

// x: (rows, d) contiguous. One work-group per row, cooperative reduction.
int candle_sycl_softmax_lastdim(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                                void *out, size_t rows, size_t d) {
  constexpr size_t WG = 256;
  return dispatch_float(dt, [&]<typename T>() -> int {
    using A = typename FAcc<T>::type;
    const T *x = static_cast<const T *>(inp);
    T *o = static_cast<T *>(out);
    try {
      q->q.submit([&](sycl::handler &h) {
        sycl::local_accessor<A, 1> red(sycl::range<1>(WG), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) {
              size_t row = it.get_group(0);
              size_t lid = it.get_local_id(0);
              const T *xr = x + row * d;
              T *orow = o + row * d;
              A lmax = -std::numeric_limits<A>::infinity();
              for (size_t j = lid; j < d; j += WG) lmax = sycl::fmax(lmax, (A)xr[j]);
              red[lid] = lmax;
              it.barrier(sycl::access::fence_space::local_space);
              for (size_t s = WG / 2; s > 0; s >>= 1) {
                if (lid < s) red[lid] = sycl::fmax(red[lid], red[lid + s]);
                it.barrier(sycl::access::fence_space::local_space);
              }
              A mx = red[0];
              it.barrier(sycl::access::fence_space::local_space);
              A lsum = 0;
              for (size_t j = lid; j < d; j += WG) {
                A e = sycl::exp((A)xr[j] - mx);
                orow[j] = (T)e;
                lsum += e;
              }
              red[lid] = lsum;
              it.barrier(sycl::access::fence_space::local_space);
              for (size_t s = WG / 2; s > 0; s >>= 1) {
                if (lid < s) red[lid] += red[lid + s];
                it.barrier(sycl::access::fence_space::local_space);
              }
              A inv = A(1) / red[0];
              for (size_t j = lid; j < d; j += WG) orow[j] = (T)((A)orow[j] * inv);
            });
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

// x: (rows, d) contiguous, alpha: (d,). d[j] = x[j] / sqrt(mean(x^2)+eps) * alpha[j].
// Sum of squares accumulates in f32 (matches candle CPU).
int candle_sycl_rms_norm(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                         const void *alpha, void *out, size_t rows, size_t d, float eps) {
  constexpr size_t WG = 256;
  return dispatch_float(dt, [&]<typename T>() -> int {
    const T *x = static_cast<const T *>(inp);
    const T *a = static_cast<const T *>(alpha);
    T *o = static_cast<T *>(out);
    try {
      q->q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(WG), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) {
              size_t row = it.get_group(0);
              size_t lid = it.get_local_id(0);
              const T *xr = x + row * d;
              T *orow = o + row * d;
              float s2 = 0.f;
              for (size_t j = lid; j < d; j += WG) {
                float v = (float)xr[j];
                s2 += v * v;
              }
              red[lid] = s2;
              it.barrier(sycl::access::fence_space::local_space);
              for (size_t s = WG / 2; s > 0; s >>= 1) {
                if (lid < s) red[lid] += red[lid + s];
                it.barrier(sycl::access::fence_space::local_space);
              }
              float m = sycl::sqrt(red[0] / (float)d + eps);
              for (size_t j = lid; j < d; j += WG) {
                orow[j] = (T)(((float)xr[j] / m) * (float)a[j]);
              }
            });
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

// mode: 0 interleaved (bhtd), 1 half-split (bhtd), 2 half-split thd (bthd).
// cos/sin: (t, d/2) or (b, t, d/2) when cos_batched.
int candle_sycl_rope(CandleSyclQueue *q, uint32_t mode, CandleSyclDType dt, const void *inp,
                     const void *cosb, const void *sinb, void *out, size_t b, size_t h,
                     size_t t, size_t d, int cos_batched) {
  size_t half = d / 2;
  size_t npairs = b * h * t * half;
  return dispatch_float(dt, [&]<typename T>() -> int {
    const T *x = static_cast<const T *>(inp);
    const T *cs = static_cast<const T *>(cosb);
    const T *sn = static_cast<const T *>(sinb);
    T *o = static_cast<T *>(out);
    try {
      q->q.parallel_for(sycl::range<1>(npairs), [=](sycl::id<1> gid) {
        size_t p = gid[0];
        size_t i1, i2, i_cs, b_i;
        if (mode == 2) {
          // src (b, t, h, d)
          size_t per_b = t * h * half;
          b_i = p / per_b;
          size_t rem = p % per_b;
          size_t i_t = rem / (h * half);
          size_t rem2 = rem % (h * half);
          size_t i_h = rem2 / half;
          size_t i_d = rem2 % half;
          size_t base = b_i * t * h * d;
          i1 = base + i_t * h * d + i_h * d + i_d;
          i2 = i1 + half;
          i_cs = i_t * half + i_d;
        } else {
          // src (b, h, t, d)
          size_t per_bh = t * half;
          size_t bh = p / per_bh;
          size_t rem = p % per_bh;
          b_i = bh / h;
          size_t i_t = rem / half;
          size_t i_d = rem % half;
          size_t base = bh * t * d;
          if (mode == 0) { // interleaved: pair (2k, 2k+1)
            size_t iov2 = i_t * half + i_d;
            i1 = base + 2 * iov2;
            i2 = i1 + 1;
            i_cs = iov2;
          } else { // half-split: pair (k, k+d/2)
            i1 = base + i_t * d + i_d;
            i2 = i1 + half;
            i_cs = i_t * half + i_d;
          }
        }
        if (cos_batched) i_cs += b_i * t * half;
        float c = (float)cs[i_cs], s = (float)sn[i_cs];
        float a = (float)x[i1], bb = (float)x[i2];
        o[i1] = (T)(a * c - bb * s);
        o[i2] = (T)(a * s + bb * c);
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

} // extern "C"
