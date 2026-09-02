// Shared helpers for the elementwise SYCL kernels.
#pragma once
#include "candle_sycl.h"
#include <sycl/sycl.hpp>
#include <cstdint>

using bf16 = sycl::ext::oneapi::bfloat16;
using f16 = sycl::half;

// Real definition of the handle that candle_sycl.h forward-declares for C.
struct CandleSyclQueue {
  sycl::queue q;
  explicit CandleSyclQueue(const sycl::device &d)
      : q(d, sycl::property::queue::in_order{}) {}
};

// Compute the flat index into a strided source for dense output position `i`.
inline int64_t strided_index(int64_t i, const CandleSyclLayout &l) {
  if (l.num_dims == 0) {
    return i;
  }
  int64_t idx = l.offset;
  for (int d = (int)l.num_dims - 1; d >= 0; --d) {
    int64_t dim = l.dims[d];
    int64_t coord = i % dim;
    i /= dim;
    idx += coord * l.strides[d];
  }
  return idx;
}

inline bool is_dense(const CandleSyclLayout &l) { return l.num_dims == 0; }

// Accumulator type for elementwise math: widen fp16/bf16 to float.
template <typename T> struct Acc {
  using type = float;
};
template <> struct Acc<double> {
  using type = double;
};
template <typename T> inline auto to_acc(T v) {
  return static_cast<typename Acc<T>::type>(v);
}

// Dispatch a templated functor over the runtime dtype. F is a generic lambda
// `[&]<typename T>() { ... }`.
template <typename F>
int dispatch_dtype(CandleSyclDType dt, F &&f) {
  switch (dt) {
  case CANDLE_SYCL_U8: return f.template operator()<uint8_t>();
  case CANDLE_SYCL_U32: return f.template operator()<uint32_t>();
  case CANDLE_SYCL_I64: return f.template operator()<int64_t>();
  case CANDLE_SYCL_F16: return f.template operator()<f16>();
  case CANDLE_SYCL_BF16: return f.template operator()<bf16>();
  case CANDLE_SYCL_F32: return f.template operator()<float>();
  case CANDLE_SYCL_F64: return f.template operator()<double>();
  default: return CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE;
  }
}

// Float-only dispatch (for ops undefined on integers).
template <typename F>
int dispatch_float(CandleSyclDType dt, F &&f) {
  switch (dt) {
  case CANDLE_SYCL_F16: return f.template operator()<f16>();
  case CANDLE_SYCL_BF16: return f.template operator()<bf16>();
  case CANDLE_SYCL_F32: return f.template operator()<float>();
  case CANDLE_SYCL_F64: return f.template operator()<double>();
  default: return CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE;
  }
}
