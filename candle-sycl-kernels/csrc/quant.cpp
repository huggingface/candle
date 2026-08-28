// GGUF block dequantization dispatch. Block layouts + per-block helpers live in
// quant_blocks.hpp (shared with mmvq.cpp).
#include "quant_blocks.hpp"

extern "C" int candle_sycl_dequantize(CandleSyclQueue *q, uint32_t ggml_dtype, const void *src,
                                      void *dst_f32, size_t n_blocks) {
  auto run = [&](auto blk_fn, int blk) -> int {
    float *y = static_cast<float *>(dst_f32);
    try {
      q->q.parallel_for(sycl::range<1>(n_blocks), [=](sycl::id<1> gid) {
        blk_fn(gid[0], y + gid[0] * blk);
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  };
  switch (ggml_dtype) {
  case G_Q4_0: { auto *b = (const BQ4_0 *)src; return run([=](size_t i, float *y){ deq_q4_0(b[i], y); }, 32); }
  case G_Q4_1: { auto *b = (const BQ4_1 *)src; return run([=](size_t i, float *y){ deq_q4_1(b[i], y); }, 32); }
  case G_Q5_0: { auto *b = (const BQ5_0 *)src; return run([=](size_t i, float *y){ deq_q5_0(b[i], y); }, 32); }
  case G_Q5_1: { auto *b = (const BQ5_1 *)src; return run([=](size_t i, float *y){ deq_q5_1(b[i], y); }, 32); }
  case G_Q8_0: { auto *b = (const BQ8_0 *)src; return run([=](size_t i, float *y){ deq_q8_0(b[i], y); }, 32); }
  case G_Q8_1: { auto *b = (const BQ8_1 *)src; return run([=](size_t i, float *y){ deq_q8_1(b[i], y); }, 32); }
  case G_Q2K: { auto *b = (const BQ2K *)src; return run([=](size_t i, float *y){ deq_q2_k(b[i], y); }, QK_K); }
  case G_Q3K: { auto *b = (const BQ3K *)src; return run([=](size_t i, float *y){ deq_q3_k(b[i], y); }, QK_K); }
  case G_Q4K: { auto *b = (const BQ4K *)src; return run([=](size_t i, float *y){ deq_q4_k(b[i], y); }, QK_K); }
  case G_Q5K: { auto *b = (const BQ5K *)src; return run([=](size_t i, float *y){ deq_q5_k(b[i], y); }, QK_K); }
  case G_Q6K: { auto *b = (const BQ6K *)src; return run([=](size_t i, float *y){ deq_q6_k(b[i], y); }, QK_K); }
  case G_Q8K: { auto *b = (const BQ8K *)src; return run([=](size_t i, float *y){ deq_q8_k(b[i], y); }, QK_K); }
  default:
    return CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE;
  }
}
