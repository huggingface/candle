// Fused quantized mat-vec (QMatMul for small batch, the decode path). The
// weight stays quantized in memory: each work-item dequantizes GGUF blocks
// inline and accumulates against the (small, <=8-row) f32 activation. Avoids
// writing/reading a 4x-inflated dequantized weight buffer — the win on a
// bandwidth-bound part like Xe-LPG. A DP4a / Q8_1 integer path is the v2.
#include "quant_blocks.hpp"

namespace {
constexpr int SG = 32;      // work-group == one sub-group
constexpr int MAX_M = 8;    // fast_mmvq batch bound

template <typename Blk, int BLK, typename Deq>
int run_mmvq(CandleSyclQueue *q, const void *w, const float *act, float *out, size_t n,
            size_t k, size_t m, Deq deq) {
  const Blk *wb = static_cast<const Blk *>(w);
  size_t nblk = k / BLK;
  try {
    q->q.submit([&](sycl::handler &h) {
      h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(n * SG), sycl::range<1>(SG)),
          [=](sycl::nd_item<1> it) {
            size_t row = it.get_group(0);
            auto sg = it.get_sub_group();
            int lid = it.get_local_id(0);
            float acc[MAX_M];
            for (int mi = 0; mi < MAX_M; ++mi) acc[mi] = 0.f;
            float blk[BLK];
            for (size_t b = lid; b < nblk; b += SG) {
              deq(wb[row * nblk + b], blk);
              for (size_t mi = 0; mi < m; ++mi) {
                const float *a = act + mi * k + b * BLK;
                float s = 0.f;
                for (int j = 0; j < BLK; ++j) s += blk[j] * a[j];
                acc[mi] += s;
              }
            }
            for (size_t mi = 0; mi < m; ++mi) {
              float total = sycl::reduce_over_group(sg, acc[mi], sycl::plus<float>());
              if (lid == 0) out[mi * n + row] = total;
            }
          });
    });
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_LAUNCH;
  }
}
} // namespace

extern "C" int candle_sycl_mmvq(CandleSyclQueue *q, uint32_t dt, const void *w, const float *act,
                                float *out, size_t n, size_t k, size_t m) {
  if (m > MAX_M) return CANDLE_SYCL_ERR_INVALID;
  switch (dt) {
  case G_Q4_0: return run_mmvq<BQ4_0, 32>(q, w, act, out, n, k, m, [](const BQ4_0 &b, float *y){ deq_q4_0(b, y); });
  case G_Q4_1: return run_mmvq<BQ4_1, 32>(q, w, act, out, n, k, m, [](const BQ4_1 &b, float *y){ deq_q4_1(b, y); });
  case G_Q5_0: return run_mmvq<BQ5_0, 32>(q, w, act, out, n, k, m, [](const BQ5_0 &b, float *y){ deq_q5_0(b, y); });
  case G_Q5_1: return run_mmvq<BQ5_1, 32>(q, w, act, out, n, k, m, [](const BQ5_1 &b, float *y){ deq_q5_1(b, y); });
  case G_Q8_0: return run_mmvq<BQ8_0, 32>(q, w, act, out, n, k, m, [](const BQ8_0 &b, float *y){ deq_q8_0(b, y); });
  case G_Q8_1: return run_mmvq<BQ8_1, 32>(q, w, act, out, n, k, m, [](const BQ8_1 &b, float *y){ deq_q8_1(b, y); });
  case G_Q2K: return run_mmvq<BQ2K, 256>(q, w, act, out, n, k, m, [](const BQ2K &b, float *y){ deq_q2_k(b, y); });
  case G_Q3K: return run_mmvq<BQ3K, 256>(q, w, act, out, n, k, m, [](const BQ3K &b, float *y){ deq_q3_k(b, y); });
  case G_Q4K: return run_mmvq<BQ4K, 256>(q, w, act, out, n, k, m, [](const BQ4K &b, float *y){ deq_q4_k(b, y); });
  case G_Q5K: return run_mmvq<BQ5K, 256>(q, w, act, out, n, k, m, [](const BQ5K &b, float *y){ deq_q5_k(b, y); });
  case G_Q6K: return run_mmvq<BQ6K, 256>(q, w, act, out, n, k, m, [](const BQ6K &b, float *y){ deq_q6_k(b, y); });
  case G_Q8K: return run_mmvq<BQ8K, 256>(q, w, act, out, n, k, m, [](const BQ8K &b, float *y){ deq_q8_k(b, y); });
  default: return CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE;
  }
}
