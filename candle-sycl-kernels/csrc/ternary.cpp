// Comparison (-> u8) and where_cond (select).
#include "common.hpp"

namespace {
enum CmpOp { C_EQ = 0, C_NE, C_LT, C_LE, C_GT, C_GE };

template <typename A> inline uint8_t cmp(uint32_t op, A a, A b) {
  switch (op) {
  case C_EQ: return a == b;
  case C_NE: return a != b;
  case C_LT: return a < b;
  case C_LE: return a <= b;
  case C_GT: return a > b;
  case C_GE: return a >= b;
  default: return 0;
  }
}

template <typename F> int dispatch_index(CandleSyclDType dt, F &&f) {
  switch (dt) {
  case CANDLE_SYCL_U8: return f.template operator()<uint8_t>();
  case CANDLE_SYCL_U32: return f.template operator()<uint32_t>();
  case CANDLE_SYCL_I64: return f.template operator()<int64_t>();
  default: return CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE;
  }
}
} // namespace

extern "C" {

int candle_sycl_cmp(CandleSyclQueue *q, uint32_t op, CandleSyclDType dt,
                    const CandleSyclLayout *lhs_l, const void *lhs,
                    const CandleSyclLayout *rhs_l, const void *rhs, void *out, size_t numel) {
  CandleSyclLayout LL = *lhs_l, RL = *rhs_l;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    const T *a = static_cast<const T *>(lhs);
    const T *b = static_cast<const T *>(rhs);
    uint8_t *o = static_cast<uint8_t *>(out);
    try {
      q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        int64_t i = gid[0];
        o[i] = cmp(op, to_acc(a[strided_index(i, LL)]), to_acc(b[strided_index(i, RL)]));
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

int candle_sycl_where(CandleSyclQueue *q, CandleSyclDType cond_dt, CandleSyclDType val_dt,
                      const CandleSyclLayout *cond_l, const void *cond,
                      const CandleSyclLayout *t_l, const void *t_vals,
                      const CandleSyclLayout *f_l, const void *f_vals, void *out, size_t numel) {
  CandleSyclLayout CL = *cond_l, TL = *t_l, FL = *f_l;
  return dispatch_index(cond_dt, [&]<typename I>() -> int {
    return dispatch_dtype(val_dt, [&]<typename T>() -> int {
      const I *c = static_cast<const I *>(cond);
      const T *tv = static_cast<const T *>(t_vals);
      const T *fv = static_cast<const T *>(f_vals);
      T *o = static_cast<T *>(out);
      try {
        q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
          int64_t i = gid[0];
          bool p = c[strided_index(i, CL)] != I(0);
          o[i] = p ? tv[strided_index(i, TL)] : fv[strided_index(i, FL)];
        });
        return CANDLE_SYCL_OK;
      } catch (...) {
        return CANDLE_SYCL_ERR_LAUNCH;
      }
    });
  });
}

} // extern "C"
