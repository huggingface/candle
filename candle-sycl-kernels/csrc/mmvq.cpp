// Fused quantized mat-vec (QMatMul for small batch, the decode path). The
// weight stays quantized in memory.
//
// Two paths:
//  - `candle_sycl_mmvq_q8` (preferred): the activation is quantized to int8
//    once per call (per-256 for K-quants, as the CPU `BlockQ8K::from_float`;
//    per-32 for Q4_0/Q8_0) and each weight block is dotted against it in
//    integer arithmetic, matching the CPU `vec_dot` numerics.
//  - `candle_sycl_mmvq`: dequantize each block to f32 and float-dot; the
//    fallback for types without an integer kernel.
//
// Both launch one work-item per (output row, chunk of `ch` weight blocks),
// which writes a partial sum to `tmp`, then one per output row to sum its
// chunks. `ch` is picked by the caller to bound `tmp`; 1 is fastest.
#include "quant_blocks.hpp"

namespace {
constexpr int MAX_M = 8; // batch bound of the float path

// `item(mi, row, b)` -> the dot of weight block `row*nblk + b` against
// activation row `mi`. Writes `out[mi*n + row]`.
template <typename Item>
void launch_chunked(sycl::queue &q, size_t n, size_t m, size_t nblk, size_t ch, float *tmp,
                    void *out, bool out_f16, Item item) {
  size_t nch = (nblk + ch - 1) / ch;
  // A 3D range rather than folding (mi, row) into one index: Xe has no hardware
  // integer divide, so recovering them with `/ n` and `% n` costs more than the
  // dot product it indexes for. `ch == 1` gets its own body so the dot is not
  // buried in a loop whose trip count the compiler cannot see.
  if (ch == 1) {
    q.parallel_for(sycl::range<3>(m, n, nch), [=](sycl::id<3> id) {
      size_t mi = id[0], row = id[1], c = id[2];
      tmp[(mi * n + row) * nch + c] = item(mi, row, c);
    });
  } else {
    q.parallel_for(sycl::range<3>(m, n, nch), [=](sycl::id<3> id) {
      size_t mi = id[0], row = id[1], c = id[2];
      size_t b1 = sycl::min((c + 1) * ch, nblk);
      float acc = 0.f;
      for (size_t b = c * ch; b < b1; ++b) acc += item(mi, row, b);
      tmp[(mi * n + row) * nch + c] = acc;
    });
  }
  q.parallel_for(sycl::range<2>(m, n), [=](sycl::id<2> id) {
    size_t mr = id[0] * n + id[1];
    float acc = 0.f;
    for (size_t c = 0; c < nch; ++c) acc += tmp[mr * nch + c];
    if (out_f16)
      static_cast<f16 *>(out)[mr] = (f16)acc;
    else
      static_cast<float *>(out)[mr] = acc;
  });
}

template <typename Blk, int BLK, typename Deq>
int run_mmvq(CandleSyclQueue *q, const void *w, const float *act, float *out, size_t n,
             size_t k, size_t m, float *tmp, size_t ch, Deq deq) {
  const Blk *wb = static_cast<const Blk *>(w);
  size_t nblk = k / BLK;
  try {
    launch_chunked(q->q, n, m, nblk, ch, tmp, out, false, [=](size_t mi, size_t row, size_t b) {
      float blk[BLK];
      deq(wb[row * nblk + b], blk);
      const float *a = act + mi * k + b * BLK;
      float s = 0.f;
      for (int j = 0; j < BLK; ++j) s += blk[j] * a[j];
      return s;
    });
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_LAUNCH;
  }
}

// ---- integer path ----------------------------------------------------------

// Quantize `m` rows of `k` f32s into `blk`-wide int8 blocks: `q8[m*k]`,
// `d8[m*nblk]` (dequant scale), and for `blk == 256` the per-32 int sums
// `s32[m*nblk*8]` the K-quant `min` terms need. `act` is f32, or f16 when
// `act_f16`, in which case it is widened per block.
int quantize_act(sycl::queue &q, int blk, const void *act, bool act_f16, size_t k, size_t m,
                 int8_t *q8, float *d8, int32_t *s32) {
  size_t nblk = k / blk;
  q.parallel_for(sycl::range<1>(m * nblk), [=](sycl::id<1> id) {
    size_t i = id[0];
    float xw[QK_K];
    const float *x;
    if (act_f16) {
      const f16 *h = static_cast<const f16 *>(act) + i * blk;
      for (int j = 0; j < blk; ++j) xw[j] = (float)h[j];
      x = xw;
    } else {
      x = static_cast<const float *>(act) + i * blk;
    }
    int8_t *y = q8 + i * blk;
    if (blk == QK_K) {
      // BlockQ8K::from_float: scale by the signed max-magnitude value.
      float amax = 0.f, max = 0.f;
      for (int j = 0; j < QK_K; ++j) {
        float ax = sycl::fabs(x[j]);
        if (amax < ax) {
          amax = ax;
          max = x[j];
        }
      }
      if (amax == 0.f) {
        d8[i] = 0.f;
        for (int j = 0; j < QK_K; ++j) y[j] = 0;
      } else {
        float iscale = -127.f / max;
        for (int j = 0; j < QK_K; ++j) {
          float v = sycl::round(iscale * x[j]);
          y[j] = (int8_t)sycl::clamp(v, -128.f, 127.f);
        }
        d8[i] = 1.f / iscale;
      }
      for (int g = 0; g < 8; ++g) {
        int32_t sum = 0;
        for (int j = 0; j < 32; ++j) sum += y[g * 32 + j];
        s32[i * 8 + g] = sum;
      }
    } else {
      // BlockQ8_0::from_float, but with an f32 scale.
      float amax = 0.f;
      for (int j = 0; j < QK; ++j) amax = sycl::fmax(amax, sycl::fabs(x[j]));
      float d = amax / 127.f;
      float id_ = d != 0.f ? 1.f / d : 0.f;
      for (int j = 0; j < QK; ++j) y[j] = (int8_t)sycl::round(x[j] * id_);
      d8[i] = d;
    }
  });
  return CANDLE_SYCL_OK;
}

// One weight block against one quantized activation block. `y` is `BLK` int8s,
// `d` its scale, `s` (K-quants only) its 8 per-32 sums.
//
// The quant bytes are read as wide words and unpacked in registers rather than
// read through the `uint8_t[]` members, which compiles to one scattered byte
// load per element. Alignment bounds the width per type: a block array starts
// at a USM device allocation (>= 16 B aligned), so member `p` of block `i` sits
// at `sizeof(Blk)*i + offsetof(p)`:
//   Q4_K  144 B block, qs @16, scales @4  -> qs is always 16 B aligned: vec4.
//   Q5_K  176 B block, qh @16, qs @48     -> both 16 B aligned: vec4.
//   Q6_K  210 B block                     -> only ever 2 B aligned: uint16.
//   Q4_0   18 B block, qs @2              -> 2 B aligned: uint16.
//   Q8_0   34 B block, qs @2              -> 2 B aligned: uint16.
// Do not widen a load past what this table allows: a misaligned vector load is
// undefined behaviour in SPIR-V, not merely slow.

inline float dot_blk(const BQ4_0 &x, const int8_t *y, float d, const int32_t *) {
  const uint16_t *q = reinterpret_cast<const uint16_t *>(x.qs); // 8 x u16 = 16 B
  int32_t sum = 0;
  for (int w = 0; w < 8; ++w) {
    uint32_t v = q[w];
    for (int b = 0; b < 2; ++b) {
      uint32_t byte = (v >> (8 * b)) & 0xFF;
      int j = 2 * w + b;
      sum += ((int)(byte & 0xF) - 8) * (int)y[j];
      sum += ((int)(byte >> 4) - 8) * (int)y[j + 16];
    }
  }
  return (float)x.d * d * (float)sum;
}

inline float dot_blk(const BQ8_0 &x, const int8_t *y, float d, const int32_t *) {
  const uint16_t *q = reinterpret_cast<const uint16_t *>(x.qs); // 16 x u16 = 32 B
  int32_t sum = 0;
  for (int w = 0; w < 16; ++w) {
    uint32_t v = q[w];
    sum += (int)(int8_t)(v & 0xFF) * (int)y[2 * w];
    sum += (int)(int8_t)((v >> 8) & 0xFF) * (int)y[2 * w + 1];
  }
  return (float)x.d * d * (float)sum;
}

inline float dot_blk(const BQ4K &x, const int8_t *y, float d, const int32_t *s) {
  const sycl::vec<uint32_t, 4> *qv = reinterpret_cast<const sycl::vec<uint32_t, 4> *>(x.qs);
  int32_t sumi = 0, summ = 0;
  for (int g = 0; g < 4; ++g) {
    const int8_t *yy = y + 64 * g;
    int32_t lo = 0, hi = 0;
    for (int v = 0; v < 2; ++v) { // 2 x 16 B covers the group's 32 quant bytes
      sycl::vec<uint32_t, 4> q4 = qv[2 * g + v];
      for (int w = 0; w < 4; ++w) {
        uint32_t vv = q4[w];
        const int8_t *ya = yy + 16 * v + 4 * w;
        const int8_t *yb = ya + 32;
        for (int b = 0; b < 4; ++b) {
          uint32_t byte = (vv >> (8 * b)) & 0xFF;
          lo += (int)(byte & 0xF) * (int)ya[b];
          hi += (int)(byte >> 4) * (int)yb[b];
        }
      }
    }
    uint8_t sc, mn;
    get_scale_min_k4(2 * g, x.scales, sc, mn);
    sumi += sc * lo;
    summ += mn * s[2 * g];
    get_scale_min_k4(2 * g + 1, x.scales, sc, mn);
    sumi += sc * hi;
    summ += mn * s[2 * g + 1];
  }
  return d * ((float)x.d * (float)sumi - (float)x.dmin * (float)summ);
}

inline float dot_blk(const BQ5K &x, const int8_t *y, float d, const int32_t *s) {
  const sycl::vec<uint32_t, 4> *qv = reinterpret_cast<const sycl::vec<uint32_t, 4> *>(x.qs);
  const sycl::vec<uint32_t, 4> *hv = reinterpret_cast<const sycl::vec<uint32_t, 4> *>(x.qh);
  int32_t sumi = 0, summ = 0;
  for (int g = 0; g < 4; ++g) {
    const int8_t *yy = y + 64 * g;
    const int sh1 = 2 * g, sh2 = 2 * g + 1;
    int32_t lo = 0, hi = 0;
    for (int v = 0; v < 2; ++v) {
      sycl::vec<uint32_t, 4> q4 = qv[2 * g + v];
      sycl::vec<uint32_t, 4> h4 = hv[v]; // qh is the same 32 B for every group
      for (int w = 0; w < 4; ++w) {
        uint32_t qq = q4[w], hh = h4[w];
        const int8_t *ya = yy + 16 * v + 4 * w;
        const int8_t *yb = ya + 32;
        for (int b = 0; b < 4; ++b) {
          uint32_t qb = (qq >> (8 * b)) & 0xFF;
          uint32_t hb = (hh >> (8 * b)) & 0xFF;
          lo += (int)((qb & 0xF) + (((hb >> sh1) & 1) << 4)) * (int)ya[b];
          hi += (int)((qb >> 4) + (((hb >> sh2) & 1) << 4)) * (int)yb[b];
        }
      }
    }
    uint8_t sc, mn;
    get_scale_min_k4(2 * g, x.scales, sc, mn);
    sumi += sc * lo;
    summ += mn * s[2 * g];
    get_scale_min_k4(2 * g + 1, x.scales, sc, mn);
    sumi += sc * hi;
    summ += mn * s[2 * g + 1];
  }
  return d * ((float)x.d * (float)sumi - (float)x.dmin * (float)summ);
}

inline float dot_blk(const BQ6K &x, const int8_t *y, float d, const int32_t *) {
  // Element p = 128*h + 32*qd + l: low 4 bits from ql[64h + l + 32*(qd&1)]
  // (nibble qd>>1), high 2 bits from qh[32h + l] >> 2*qd; scale per 16.
  int32_t sumi = 0;
  for (int h = 0; h < 2; ++h) {
    const uint8_t *ql = x.ql + 64 * h;
    const uint8_t *qh = x.qh + 32 * h;
    for (int qd = 0; qd < 4; ++qd) {
      const uint8_t *qlo = ql + 32 * (qd & 1);
      const int nsh = 4 * (qd >> 1), hsh = 2 * qd;
      const int8_t *yy = y + 128 * h + 32 * qd;
      int32_t a = 0, b = 0;
      for (int l = 0; l < 16; ++l)
        a += ((int)(((qlo[l] >> nsh) & 0xF) | (((qh[l] >> hsh) & 3) << 4)) - 32) * yy[l];
      for (int l = 16; l < 32; ++l)
        b += ((int)(((qlo[l] >> nsh) & 0xF) | (((qh[l] >> hsh) & 3) << 4)) - 32) * yy[l];
      sumi += x.scales[8 * h + 2 * qd] * a + x.scales[8 * h + 2 * qd + 1] * b;
    }
  }
  return (float)x.d * d * (float)sumi;
}

template <typename Blk, int BLK, typename Dot>
int run_mmvq_q8(CandleSyclQueue *cq, const void *w, const void *act, bool act_f16, void *out,
                bool out_f16, size_t n, size_t k, size_t m, int8_t *q8, float *d8, int32_t *s32,
                float *tmp, size_t ch, Dot dot) {
  const Blk *wb = static_cast<const Blk *>(w);
  size_t nblk = k / BLK;
  try {
    quantize_act(cq->q, BLK, act, act_f16, k, m, q8, d8, s32);
    launch_chunked(cq->q, n, m, nblk, ch, tmp, out, out_f16, [=](size_t mi, size_t row, size_t b) {
      size_t yb = mi * nblk + b;
      return dot(wb[row * nblk + b], q8 + yb * BLK, d8[yb], s32 + yb * 8);
    });
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_LAUNCH;
  }
}
} // namespace

// Integer mat-vec. `act` / `out` are f32, or f16 when `act_f16` / `out_f16`.
// Scratch: `q8` holds `m*k` bytes, `d8` `m*(k/blk)` f32s, `s32` `m*(k/blk)*8`
// i32s (blk = 256 for K-quants, 32 otherwise), `tmp` `m*n*ceil((k/blk)/ch)`
// f32s. Returns CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE for types without an integer
// kernel; the caller then uses `candle_sycl_mmvq`.
extern "C" int candle_sycl_mmvq_q8(CandleSyclQueue *q, uint32_t dt, const void *w,
                                   const void *act, int act_f16, void *out, int out_f16,
                                   size_t n, size_t k, size_t m, int8_t *q8, float *d8,
                                   int32_t *s32, float *tmp, size_t ch) {
  const bool af = act_f16 != 0, of = out_f16 != 0;
  // `dot_blk` is overloaded on the block type, so one lambda serves every case.
  auto dot = [](const auto &x, const int8_t *y, float d, const int32_t *s) {
    return dot_blk(x, y, d, s);
  };
  switch (dt) {
  case G_Q4_0: return run_mmvq_q8<BQ4_0, QK>(q, w, act, af, out, of, n, k, m, q8, d8, s32, tmp, ch, dot);
  case G_Q8_0: return run_mmvq_q8<BQ8_0, QK>(q, w, act, af, out, of, n, k, m, q8, d8, s32, tmp, ch, dot);
  case G_Q4K: return run_mmvq_q8<BQ4K, QK_K>(q, w, act, af, out, of, n, k, m, q8, d8, s32, tmp, ch, dot);
  case G_Q5K: return run_mmvq_q8<BQ5K, QK_K>(q, w, act, af, out, of, n, k, m, q8, d8, s32, tmp, ch, dot);
  case G_Q6K: return run_mmvq_q8<BQ6K, QK_K>(q, w, act, af, out, of, n, k, m, q8, d8, s32, tmp, ch, dot);
  default: return CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE;
  }
}

// Float mat-vec; `tmp` as for `candle_sycl_mmvq_q8`.
extern "C" int candle_sycl_mmvq(CandleSyclQueue *q, uint32_t dt, const void *w, const float *act,
                                float *out, size_t n, size_t k, size_t m, float *tmp,
                                size_t ch) {
  if (m > MAX_M) return CANDLE_SYCL_ERR_INVALID;
  return dispatch_blk(dt, [&](auto tag) {
    return run_mmvq<typename decltype(tag)::Blk, decltype(tag)::blk>(
        q, w, act, out, n, k, m, tmp, ch,
        [](const auto &b, float *y) { deq_blk(b, y); });
  });
}
