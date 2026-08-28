// Indexing kernels: index_select, gather, scatter(_add), index_add, and a
// per-row bitonic argsort. `inp`/`out`/`ids` are contiguous unless noted;
// index_select additionally accepts a strided source via `lin`.
#include "common.hpp"
#include <limits>
#include <type_traits>

namespace {

template <typename F> int dispatch_index(CandleSyclDType dt, F &&f) {
  switch (dt) {
  case CANDLE_SYCL_U8: return f.template operator()<uint8_t>();
  case CANDLE_SYCL_U32: return f.template operator()<uint32_t>();
  case CANDLE_SYCL_I64: return f.template operator()<int64_t>();
  default: return CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE;
  }
}

template <typename I> inline size_t as_index(I v) { return (size_t)(int64_t)v; }

// candle uses the max value of the index type as a "skip / write zero" sentinel.
template <typename I> inline bool is_skip(I v) {
  if constexpr (std::is_same_v<I, int64_t>) {
    return v == (int64_t)0x7FFFFFFFFFFFFFFFLL;
  } else {
    return v == std::numeric_limits<I>::max();
  }
}

inline size_t next_pow2(size_t n) {
  size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

} // namespace

extern "C" {

// out[left, ids_dim, right] = inp[left, ids[id], right]
int candle_sycl_index_select(CandleSyclQueue *q, CandleSyclDType dt, CandleSyclDType idt,
                             const CandleSyclLayout *lin, const void *inp, const void *ids,
                             void *out, size_t left_size, size_t src_dim_size,
                             size_t ids_dim_size, size_t right_size) {
  CandleSyclLayout L = *lin;
  size_t numel = left_size * ids_dim_size * right_size;
  return dispatch_index(idt, [&]<typename I>() -> int {
    return dispatch_dtype(dt, [&]<typename T>() -> int {
      const T *in = static_cast<const T *>(inp);
      const I *id = static_cast<const I *>(ids);
      T *o = static_cast<T *>(out);
      try {
        q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
          size_t dst_i = gid[0];
          size_t right_i = dst_i % right_size;
          size_t id_i = (dst_i / right_size) % ids_dim_size;
          size_t left_i = dst_i / (ids_dim_size * right_size);
          I raw = id[id_i];
          if (is_skip(raw)) { o[dst_i] = T(0); return; }
          size_t j = as_index(raw);
          size_t src_i = left_i * src_dim_size * right_size + j * right_size + right_i;
          o[dst_i] = in[strided_index((int64_t)src_i, L)];
        });
        return CANDLE_SYCL_OK;
      } catch (...) {
        return CANDLE_SYCL_ERR_LAUNCH;
      }
    });
  });
}

// out[i] = inp[ pre*src_dim*right + ids[i]*right + post ], ids shape == out shape
int candle_sycl_gather(CandleSyclQueue *q, CandleSyclDType dt, CandleSyclDType idt,
                       const CandleSyclLayout *lin, const void *inp, const void *ids,
                       void *out, size_t left_size, size_t src_dim_size,
                       size_t ids_dim_size, size_t right_size) {
  CandleSyclLayout L = *lin;
  size_t numel = left_size * ids_dim_size * right_size;
  return dispatch_index(idt, [&]<typename I>() -> int {
    return dispatch_dtype(dt, [&]<typename T>() -> int {
      const T *in = static_cast<const T *>(inp);
      const I *id = static_cast<const I *>(ids);
      T *o = static_cast<T *>(out);
      try {
        q->q.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
          size_t i = gid[0];
          size_t post = i % right_size;
          size_t pre = i / (right_size * ids_dim_size);
          I raw = id[i];
          if (is_skip(raw)) { o[i] = T(0); return; }
          size_t j = as_index(raw);
          size_t src_i = (pre * src_dim_size + j) * right_size + post;
          o[i] = in[strided_index((int64_t)src_i, L)];
        });
        return CANDLE_SYCL_OK;
      } catch (...) {
        return CANDLE_SYCL_ERR_LAUNCH;
      }
    });
  });
}

// in-place: out[pre, ids[k], post] {=|+=} src[pre, k, post] for k in src_dim
int candle_sycl_scatter(CandleSyclQueue *q, int add, CandleSyclDType dt, CandleSyclDType idt,
                        void *out, const void *ids, const void *src, size_t left_size,
                        size_t src_dim_size, size_t dst_dim_size, size_t right_size) {
  size_t outer = left_size * right_size;
  return dispatch_index(idt, [&]<typename I>() -> int {
    return dispatch_dtype(dt, [&]<typename T>() -> int {
      T *o = static_cast<T *>(out);
      const I *id = static_cast<const I *>(ids);
      const T *s = static_cast<const T *>(src);
      try {
        q->q.parallel_for(sycl::range<1>(outer), [=](sycl::id<1> gid) {
          size_t i = gid[0];
          size_t pre = i / right_size;
          size_t post = i % right_size;
          for (size_t k = 0; k < src_dim_size; ++k) {
            size_t si = (pre * src_dim_size + k) * right_size + post;
            if (is_skip(id[si])) continue;
            size_t di = (pre * dst_dim_size + as_index(id[si])) * right_size + post;
            if (add) {
              o[di] = static_cast<T>(to_acc(o[di]) + to_acc(s[si]));
            } else {
              o[di] = s[si];
            }
          }
        });
        return CANDLE_SYCL_OK;
      } catch (...) {
        return CANDLE_SYCL_ERR_LAUNCH;
      }
    });
  });
}

// in-place: out[pre, ids[j], post] += src[pre, j, post] for j in ids_dim
int candle_sycl_index_add(CandleSyclQueue *q, CandleSyclDType dt, CandleSyclDType idt, void *out,
                          const void *ids, const void *src, size_t left_size,
                          size_t ids_dim_size, size_t dst_dim_size, size_t right_size) {
  size_t outer = left_size * right_size;
  return dispatch_index(idt, [&]<typename I>() -> int {
    return dispatch_dtype(dt, [&]<typename T>() -> int {
      T *o = static_cast<T *>(out);
      const I *id = static_cast<const I *>(ids);
      const T *s = static_cast<const T *>(src);
      try {
        q->q.parallel_for(sycl::range<1>(outer), [=](sycl::id<1> gid) {
          size_t i = gid[0];
          size_t pre = i / right_size;
          size_t post = i % right_size;
          for (size_t j = 0; j < ids_dim_size; ++j) {
            if (is_skip(id[j])) continue;
            size_t si = (pre * ids_dim_size + j) * right_size + post;
            size_t di = (pre * dst_dim_size + as_index(id[j])) * right_size + post;
            o[di] = static_cast<T>(to_acc(o[di]) + to_acc(s[si]));
          }
        });
        return CANDLE_SYCL_OK;
      } catch (...) {
        return CANDLE_SYCL_ERR_LAUNCH;
      }
    });
  });
}

// Per-row argsort along the last dim. One work-group per row, bitonic sort of
// `ncols_pad` (next pow2) index slots in local memory.
int candle_sycl_argsort(CandleSyclQueue *q, CandleSyclDType dt, int ascending, const void *inp,
                        uint32_t *out, size_t nrows, size_t ncols) {
  size_t ncols_pad = next_pow2(ncols);
  size_t threads = ncols_pad < 1024 ? ncols_pad : 1024;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    const T *in = static_cast<const T *>(inp);
    try {
      q->q.submit([&](sycl::handler &h) {
        sycl::local_accessor<int32_t, 1> slot(sycl::range<1>(ncols_pad), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nrows * threads), sycl::range<1>(threads)),
            [=](sycl::nd_item<1> it) {
              size_t row = it.get_group(0);
              size_t lid = it.get_local_id(0);
              const T *x = in + row * ncols;
              for (size_t c = lid; c < ncols_pad; c += threads) {
                slot[c] = c < ncols ? (int32_t)c : -1;
              }
              it.barrier(sycl::access::fence_space::local_space);
              auto less = [&](int a, int b) {
                if (a < 0) return false; // pad sinks to the end
                if (b < 0) return true;
                T va = x[a], vb = x[b];
                return ascending ? (va < vb) : (va > vb);
              };
              for (size_t k = 2; k <= ncols_pad; k <<= 1) {
                for (size_t j = k >> 1; j > 0; j >>= 1) {
                  for (size_t c = lid; c < ncols_pad; c += threads) {
                    size_t ixj = c ^ j;
                    if (ixj > c) {
                      bool up = ((c & k) == 0);
                      int a = slot[c], b = slot[ixj];
                      // ascending block: smaller (per `less`) belongs at `c`.
                      bool swap = up ? less(b, a) : less(a, b);
                      if (swap) {
                        slot[c] = b;
                        slot[ixj] = a;
                      }
                    }
                  }
                  it.barrier(sycl::access::fence_space::local_space);
                }
              }
              for (size_t c = lid; c < ncols; c += threads) {
                out[row * ncols + c] = (uint32_t)slot[c];
              }
            });
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}

} // extern "C"
