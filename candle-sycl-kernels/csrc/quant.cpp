// GGUF dequantization kernels. Block layouts and per-block helpers live in
// quant_blocks.hpp (shared with mmvq.cpp).
#include "quant_blocks.hpp"

namespace {
// `src_blk(i)` -> the source block for destination block `i`, so a plain
// dequantize and a gather share one body. `T` is the output element type; the
// block arithmetic is float either way.
template <typename T, typename Map>
int dequantize_into(CandleSyclQueue *q, uint32_t ggml_dtype, const void *src, void *dst,
                    size_t n_blocks, Map src_blk) {
  return dispatch_blk(ggml_dtype, [&](auto tag) {
    using Blk = typename decltype(tag)::Blk;
    const Blk *b = static_cast<const Blk *>(src);
    T *y = static_cast<T *>(dst);
    try {
      q->q.parallel_for(sycl::range<1>(n_blocks), [=](sycl::id<1> gid) {
        size_t i = gid[0];
        deq_blk(b[src_blk(i)], y + i * decltype(tag)::blk);
      });
      return CANDLE_SYCL_OK;
    } catch (...) {
      return CANDLE_SYCL_ERR_LAUNCH;
    }
  });
}
} // namespace

extern "C" int candle_sycl_dequantize(CandleSyclQueue *q, uint32_t ggml_dtype, const void *src,
                                      void *dst_f32, size_t n_blocks) {
  return dequantize_into<float>(q, ggml_dtype, src, dst_f32, n_blocks,
                                [](size_t i) { return i; });
}

// As above but writes f16, for a half-precision pipeline that would otherwise
// dequantize to f32 and cast.
extern "C" int candle_sycl_dequantize_f16(CandleSyclQueue *q, uint32_t ggml_dtype, const void *src,
                                          void *dst_f16, size_t n_blocks) {
  return dequantize_into<f16>(q, ggml_dtype, src, dst_f16, n_blocks, [](size_t i) { return i; });
}

// Gather-and-dequantize whole rows of a quantized `(rows, row_blocks * blk)`
// matrix: `dst[i] = dequant(src row ids[i])`. The embedding path — touches only
// the selected rows instead of dequantizing the whole table.
extern "C" int candle_sycl_get_rows(CandleSyclQueue *q, uint32_t ggml_dtype, const void *src,
                                    const uint32_t *ids, void *dst_f32, size_t n_ids,
                                    size_t row_blocks) {
  return dequantize_into<float>(q, ggml_dtype, src, dst_f32, n_ids * row_blocks, [=](size_t i) {
    return (size_t)ids[i / row_blocks] * row_blocks + i % row_blocks;
  });
}
