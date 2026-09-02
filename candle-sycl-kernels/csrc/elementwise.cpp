// Elementwise SYCL kernels: fill, affine, unary, binary, cast, strided copy.
#include "common.hpp"
#include <cmath>

namespace {

constexpr size_t WG = 256;

inline size_t rounded(size_t n) { return ((n + WG - 1) / WG) * WG; }

// ---- unary op table --------------------------------------------------------
enum UnaryOp {
  U_COPY = 0, U_NEG, U_ABS, U_SQR, U_SQRT, U_RECIP, U_EXP, U_LOG, U_SIN, U_COS,
  U_TANH, U_ERF, U_CEIL, U_FLOOR, U_ROUND, U_SIGN, U_RELU, U_SILU, U_GELU,
  U_GELU_ERF, U_SIGMOID,
};

template <typename A> inline A apply_unary(uint32_t op, A x) {
  switch (op) {
  case U_COPY: return x;
  case U_NEG: return -x;
  case U_ABS: return sycl::fabs(x);
  case U_SQR: return x * x;
  case U_SQRT: return sycl::sqrt(x);
  case U_RECIP: return A(1) / x;
  case U_EXP: return sycl::exp(x);
  case U_LOG: return sycl::log(x);
  case U_SIN: return sycl::sin(x);
  case U_COS: return sycl::cos(x);
  case U_TANH: return sycl::tanh(x);
  case U_ERF: return sycl::erf(x);
  case U_CEIL: return sycl::ceil(x);
  case U_FLOOR: return sycl::floor(x);
  case U_ROUND: return sycl::round(x);
  case U_SIGN: return A((x > A(0)) - (x < A(0)));
  case U_RELU: return sycl::fmax(x, A(0));
  case U_SILU: return x / (A(1) + sycl::exp(-x));
  case U_GELU: {
    const A c = A(0.7978845608028654); // sqrt(2/pi)
    A inner = c * (x + A(0.044715) * x * x * x);
    return A(0.5) * x * (A(1) + sycl::tanh(inner));
  }
  case U_GELU_ERF:
    return A(0.5) * x * (A(1) + sycl::erf(x * A(0.7071067811865476)));
  case U_SIGMOID: return A(1) / (A(1) + sycl::exp(-x));
  default: return x;
  }
}

enum BinaryOp { B_ADD = 0, B_SUB, B_MUL, B_DIV, B_MAX, B_MIN };

template <typename A> inline A apply_binary(uint32_t op, A a, A b) {
  switch (op) {
  case B_ADD: return a + b;
  case B_SUB: return a - b;
  case B_MUL: return a * b;
  case B_DIV: return a / b;
  case B_MAX: return sycl::fmax(a, b);
  case B_MIN: return sycl::fmin(a, b);
  default: return a;
  }
}

} // namespace

extern "C" {

int candle_sycl_fill(CandleSyclQueue *q, CandleSyclDType dt, void *dst, size_t numel,
                     double value) {
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    T *p = static_cast<T *>(dst);
    T v = static_cast<T>(value);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) { p[i] = v; });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

int candle_sycl_fill_strided(CandleSyclQueue *q, CandleSyclDType dt,
                             const CandleSyclLayout *lin, void *dst, size_t numel,
                             double value) {
  CandleSyclLayout L = *lin;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    T *p = static_cast<T *>(dst);
    T v = static_cast<T>(value);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
        p[strided_index((int64_t)i[0], L)] = v;
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

int candle_sycl_affine(CandleSyclQueue *q, CandleSyclDType dt, const CandleSyclLayout *lin,
                       const void *inp, void *out, size_t numel, double mul, double add) {
  CandleSyclLayout L = *lin;
  return dispatch_float(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    T *o = static_cast<T *>(out);
    using A = typename Acc<T>::type;
    A m = (A)mul, a = (A)add;
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        int64_t i = gid[0];
        int64_t si = strided_index(i, L);
        o[i] = static_cast<T>(to_acc(in[si]) * m + a);
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

int candle_sycl_elu(CandleSyclQueue *q, CandleSyclDType dt, const CandleSyclLayout *lin,
                    const void *inp, void *out, size_t numel, double alpha) {
  CandleSyclLayout L = *lin;
  return dispatch_float(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    T *o = static_cast<T *>(out);
    using A = typename Acc<T>::type;
    A a = (A)alpha;
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        int64_t i = gid[0];
        A x = to_acc(in[strided_index(i, L)]);
        o[i] = static_cast<T>(x > A(0) ? x : a * (sycl::exp(x) - A(1)));
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

int candle_sycl_powf(CandleSyclQueue *q, CandleSyclDType dt, const CandleSyclLayout *lin,
                     const void *inp, void *out, size_t numel, double exponent) {
  CandleSyclLayout L = *lin;
  return dispatch_float(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    T *o = static_cast<T *>(out);
    using A = typename Acc<T>::type;
    A e = (A)exponent;
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        int64_t i = gid[0];
        o[i] = static_cast<T>(sycl::pow(to_acc(in[strided_index(i, L)]), e));
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

int candle_sycl_unary(CandleSyclQueue *q, uint32_t op, CandleSyclDType dt,
                      const CandleSyclLayout *lin, const void *inp, void *out,
                      size_t numel) {
  CandleSyclLayout L = *lin;
  return dispatch_float(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    T *o = static_cast<T *>(out);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        int64_t i = gid[0];
        int64_t si = strided_index(i, L);
        o[i] = static_cast<T>(apply_unary(op, to_acc(in[si])));
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

int candle_sycl_binary(CandleSyclQueue *q, uint32_t op, CandleSyclDType dt,
                       const CandleSyclLayout *lhs_l, const void *lhs,
                       const CandleSyclLayout *rhs_l, const void *rhs, void *out,
                       size_t numel) {
  CandleSyclLayout LL = *lhs_l, RL = *rhs_l;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    const T *a = static_cast<const T *>(lhs);
    const T *b = static_cast<const T *>(rhs);
    T *o = static_cast<T *>(out);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        int64_t i = gid[0];
        int64_t ai = strided_index(i, LL);
        int64_t bi = strided_index(i, RL);
        o[i] = static_cast<T>(apply_binary(op, to_acc(a[ai]), to_acc(b[bi])));
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

int candle_sycl_cast(CandleSyclQueue *q, CandleSyclDType src_dt, CandleSyclDType dst_dt,
                     const CandleSyclLayout *lin, const void *inp, void *out,
                     size_t numel) {
  CandleSyclLayout L = *lin;
  return dispatch_dtype(src_dt, [&]<typename S>() -> int {
    return dispatch_dtype(dst_dt, [&]<typename D>() -> int {
      const S *in = static_cast<const S *>(inp);
      D *o = static_cast<D *>(out);
      try {
        q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
          int64_t i = gid[0];
          int64_t si = strided_index(i, L);
          o[i] = static_cast<D>(static_cast<double>(in[si]));
        });
        return CANDLE_SYCL_OK;
      } catch (...) {
        return CANDLE_SYCL_ERR_LAUNCH;
      }
    });
  });
}

int candle_sycl_copy_strided(CandleSyclQueue *q, CandleSyclDType dt,
                             const CandleSyclLayout *lin, const void *inp, void *out,
                             size_t dst_offset, size_t numel) {
  CandleSyclLayout L = *lin;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    T *o = static_cast<T *>(out) + dst_offset;
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        int64_t i = gid[0];
        o[i] = in[strided_index(i, L)];
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

int candle_sycl_copy2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp, void *out,
                       size_t d1, size_t d2, size_t src_stride1, size_t dst_stride1,
                       size_t src_offset, size_t dst_offset) {
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp) + src_offset;
    T *o = static_cast<T *>(out) + dst_offset;
    try {
      q->q.parallel_for(sycl::range<2>(d1, d2), [=](sycl::id<2> id) {
        size_t i = id[0], j = id[1];
        o[i * dst_stride1 + j] = in[i * src_stride1 + j];
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

} // extern "C"
