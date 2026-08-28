// Reductions: sum / min / max / argmin / argmax over a set of trailing
// (post-reorder) dims. `lin` describes the source with the reduced dims moved
// last, so the flat index `out_i * reduce_el + r` decomposes directly.
// One work-group per output element; work-items cooperatively reduce.
#include "common.hpp"
#include <limits>

namespace {

enum ReduceOp { R_SUM = 0, R_MIN = 1, R_MAX = 2, R_ARGMIN = 3, R_ARGMAX = 4 };

template <typename T> struct RAcc {
  using type = float;
};
template <> struct RAcc<double> {
  using type = double;
};
template <> struct RAcc<int64_t> {
  using type = int64_t;
};
template <> struct RAcc<uint32_t> {
  using type = int64_t;
};
template <> struct RAcc<uint8_t> {
  using type = int64_t;
};

constexpr size_t WG = 256;

} // namespace

extern "C" int candle_sycl_reduce(CandleSyclQueue *q, uint32_t op, CandleSyclDType dt,
                                  const CandleSyclLayout *lin, const void *inp, void *out,
                                  size_t out_el, size_t reduce_el) {
  CandleSyclLayout L = *lin;
  return dispatch_dtype(dt, [&]<typename T>() -> int {
    using A = typename RAcc<T>::type;
    const T *in = static_cast<const T *>(inp);
    bool want_index = (op == R_ARGMIN || op == R_ARGMAX);
    bool want_min = (op == R_MIN || op == R_ARGMIN);
    T *o_val = static_cast<T *>(out);
    uint32_t *o_idx = static_cast<uint32_t *>(out);
    // A contiguous source is the common case — skip the per-element index math.
    bool dense = (L.num_dims == 0);
    try {
      q->q.submit([&](sycl::handler &h) {
        sycl::local_accessor<T, 1> lv(sycl::range<1>(WG), h);
        sycl::local_accessor<uint32_t, 1> li(sycl::range<1>(WG), h);
        sycl::local_accessor<A, 1> ls(sycl::range<1>(WG), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(out_el * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) {
              size_t i = it.get_group(0);
              size_t lid = it.get_local_id(0);
              size_t base = i * reduce_el;

              if (op == R_SUM) {
                A acc = 0;
                for (size_t r = lid; r < reduce_el; r += WG) {
                  size_t si = dense ? base + r : strided_index((int64_t)(base + r), L);
                  acc += static_cast<A>(in[si]);
                }
                ls[lid] = acc;
                it.barrier(sycl::access::fence_space::local_space);
                for (size_t s = WG / 2; s > 0; s >>= 1) {
                  if (lid < s) ls[lid] += ls[lid + s];
                  it.barrier(sycl::access::fence_space::local_space);
                }
                if (lid == 0) o_val[i] = static_cast<T>(ls[0]);
                return;
              }

              // min / max / arg*: manual local-memory tree reduction (group
              // reductions over half/bf16/i64 min/max are not portable).
              T ident = want_min ? std::numeric_limits<T>::max()
                                 : std::numeric_limits<T>::lowest();
              T best = ident;
              uint32_t best_r = 0;
              for (size_t r = lid; r < reduce_el; r += WG) {
                size_t si = dense ? base + r : strided_index((int64_t)(base + r), L);
                T v = in[si];
                if (want_min ? (v < best) : (v > best)) {
                  best = v;
                  best_r = (uint32_t)r;
                }
              }
              lv[lid] = best;
              li[lid] = best_r;
              it.barrier(sycl::access::fence_space::local_space);
              for (size_t s = WG / 2; s > 0; s >>= 1) {
                if (lid < s) {
                  T o = lv[lid + s];
                  bool take = want_min ? (o < lv[lid]) : (o > lv[lid]);
                  if (take || (o == lv[lid] && li[lid + s] < li[lid])) {
                    lv[lid] = o;
                    li[lid] = li[lid + s];
                  }
                }
                it.barrier(sycl::access::fence_space::local_space);
              }
              if (lid == 0) {
                if (want_index) {
                  o_idx[i] = li[0];
                } else {
                  o_val[i] = lv[0];
                }
              }
            });
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}
