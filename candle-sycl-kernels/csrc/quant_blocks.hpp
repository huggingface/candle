// Shared GGUF block layouts + per-block dequant helpers (candle CPU `to_float`).
#pragma once
#include "common.hpp"
#include <cstring>

#define QK 32
#define QK_K 256
#define K_SCALE_SIZE 12

#pragma pack(push, 1)
struct BQ4_0 { f16 d; uint8_t qs[16]; };
struct BQ4_1 { f16 d, m; uint8_t qs[16]; };
struct BQ5_0 { f16 d; uint8_t qh[4]; uint8_t qs[16]; };
struct BQ5_1 { f16 d, m; uint8_t qh[4]; uint8_t qs[16]; };
struct BQ8_0 { f16 d; int8_t qs[32]; };
struct BQ8_1 { f16 d, s; int8_t qs[32]; };
struct BQ2K { uint8_t scales[16]; uint8_t qs[64]; f16 d, dmin; };
struct BQ3K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; f16 d; };
struct BQ4K { f16 d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[128]; };
struct BQ5K { f16 d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[32]; uint8_t qs[128]; };
struct BQ6K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; f16 d; };
struct BQ8K { float d; int8_t qs[256]; int16_t bsums[16]; };
#pragma pack(pop)

// Kernel-local dtype ids (candle source order). The Rust side maps
// `quantized::GgmlDType` -> these; see `GgmlDType` in src/lib.rs.
enum {
  G_F32 = 0, G_F16, G_BF16, G_Q4_0, G_Q4_1, G_Q5_0, G_Q5_1, G_Q8_0, G_Q8_1,
  G_Q2K, G_Q3K, G_Q4K, G_Q5K, G_Q6K, G_Q8K,
};


inline uint32_t rd_u32(const uint8_t *p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t &d, uint8_t &m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// -- one block -> `qk`/QK_K values at y --
// Overloaded on the block type, and templated on the output element type: the
// block arithmetic is always float, only the store converts, so a
// half-precision pipeline can dequantize straight to f16 rather than
// materialising f32 and casting.
template <typename T>
inline void deq_blk(const BQ4_0 &b, T *y) {
  float d = (float)b.d;
  for (int j = 0; j < 16; ++j) {
    y[j] = ((int)(b.qs[j] & 0xF) - 8) * d;
    y[j + 16] = ((int)(b.qs[j] >> 4) - 8) * d;
  }
}
template <typename T>
inline void deq_blk(const BQ4_1 &b, T *y) {
  float d = (float)b.d, m = (float)b.m;
  for (int j = 0; j < 16; ++j) {
    y[j] = (b.qs[j] & 0xF) * d + m;
    y[j + 16] = (b.qs[j] >> 4) * d + m;
  }
}
template <typename T>
inline void deq_blk(const BQ5_0 &b, T *y) {
  float d = (float)b.d;
  uint32_t qh = rd_u32(b.qh);
  for (int j = 0; j < 16; ++j) {
    uint8_t xh0 = ((qh >> j) << 4) & 0x10;
    uint8_t xh1 = (qh >> (j + 12)) & 0x10;
    y[j] = (((int)((b.qs[j] & 0xF) | xh0)) - 16) * d;
    y[j + 16] = (((int)((b.qs[j] >> 4) | xh1)) - 16) * d;
  }
}
template <typename T>
inline void deq_blk(const BQ5_1 &b, T *y) {
  float d = (float)b.d, m = (float)b.m;
  uint32_t qh = rd_u32(b.qh);
  for (int j = 0; j < 16; ++j) {
    uint8_t xh0 = ((qh >> j) << 4) & 0x10;
    uint8_t xh1 = (qh >> (j + 12)) & 0x10;
    y[j] = (float)((b.qs[j] & 0xF) | xh0) * d + m;
    y[j + 16] = (float)((b.qs[j] >> 4) | xh1) * d + m;
  }
}
template <typename T>
inline void deq_blk(const BQ8_0 &b, T *y) {
  float d = (float)b.d;
  for (int j = 0; j < 32; ++j) y[j] = b.qs[j] * d;
}
template <typename T>
inline void deq_blk(const BQ8_1 &b, T *y) {
  float d = (float)b.d;
  for (int j = 0; j < 32; ++j) y[j] = b.qs[j] * d;
}

template <typename T>
inline void deq_blk(const BQ2K &b, T *y) {
  float d = (float)b.d, mn = (float)b.dmin;
  int is = 0;
  for (int blk = 0; blk < 2; ++blk) {
    const uint8_t *qs = b.qs + 32 * blk;
    int yi = 128 * blk;
    for (int shift = 0; shift < 8; shift += 2) {
      uint8_t sc = b.scales[is++];
      float dl = d * (sc & 0xF), ml = mn * (sc >> 4);
      for (int q = 0; q < 16; ++q)
        y[yi++] = dl * ((qs[q] >> shift) & 3) - ml;
      sc = b.scales[is++];
      dl = d * (sc & 0xF);
      ml = mn * (sc >> 4);
      for (int q = 16; q < 32; ++q)
        y[yi++] = dl * ((qs[q] >> shift) & 3) - ml;
    }
  }
}

template <typename T>
inline void deq_blk(const BQ3K &b, T *y) {
  const uint32_t KM1 = 0x03030303, KM2 = 0x0f0f0f0f;
  uint32_t aux[4] = {0, 0, 0, 0};
  std::memcpy(aux, b.scales, 12);
  uint32_t tmp = aux[2];
  aux[2] = ((aux[0] >> 4) & KM2) | (((tmp >> 4) & KM1) << 4);
  aux[3] = ((aux[1] >> 4) & KM2) | (((tmp >> 6) & KM1) << 4);
  aux[0] = (aux[0] & KM2) | (((tmp)&KM1) << 4);
  aux[1] = (aux[1] & KM2) | (((tmp >> 2) & KM1) << 4);
  const int8_t *scales = (const int8_t *)aux;
  float d_all = (float)b.d;
  uint8_t m = 1;
  int is = 0, yi = 0;
  for (int j128 = 0; j128 < 2; ++j128) {
    const uint8_t *qs = b.qs + 32 * j128;
    for (int shift = 0; shift < 8; shift += 2) {
      for (int si = 0; si < 2; ++si) {
        float dl = d_all * (scales[is] - 32.f);
        for (int i = 0; i < 16; ++i) {
          int hm = (b.hmask[i + 16 * si] & m) ? 0 : 4;
          y[yi++] = dl * (float)(((qs[i + 16 * si] >> shift) & 3) - hm);
        }
        is++;
      }
      m <<= 1;
    }
  }
}

template <typename T>
inline void deq_blk(const BQ4K &b, T *y) {
  float d = (float)b.d, mn = (float)b.dmin;
  int is = 0, yi = 0;
  for (int j = 0; j < QK_K; j += 64) {
    const uint8_t *q = b.qs + j / 2;
    uint8_t sc, m;
    get_scale_min_k4(is, b.scales, sc, m);
    float d1 = d * sc, m1 = mn * m;
    get_scale_min_k4(is + 1, b.scales, sc, m);
    float d2 = d * sc, m2 = mn * m;
    for (int l = 0; l < 32; ++l) y[yi++] = d1 * (q[l] & 0xF) - m1;
    for (int l = 0; l < 32; ++l) y[yi++] = d2 * (q[l] >> 4) - m2;
    is += 2;
  }
}

template <typename T>
inline void deq_blk(const BQ5K &b, T *y) {
  float d = (float)b.d, mn = (float)b.dmin;
  int is = 0, yi = 0;
  uint8_t u1 = 1, u2 = 2;
  for (int j = 0; j < QK_K; j += 64) {
    const uint8_t *ql = b.qs + j / 2;
    uint8_t sc, m;
    get_scale_min_k4(is, b.scales, sc, m);
    float d1 = d * sc, m1 = mn * m;
    get_scale_min_k4(is + 1, b.scales, sc, m);
    float d2 = d * sc, m2 = mn * m;
    for (int l = 0; l < 32; ++l) {
      float add = (b.qh[l] & u1) ? 16.f : 0.f;
      y[yi++] = d1 * ((ql[l] & 0xF) + add) - m1;
    }
    for (int l = 0; l < 32; ++l) {
      float add = (b.qh[l] & u2) ? 16.f : 0.f;
      y[yi++] = d2 * ((ql[l] >> 4) + add) - m2;
    }
    is += 2;
    u1 <<= 2;
    u2 <<= 2;
  }
}

template <typename T>
inline void deq_blk(const BQ6K &b, T *y) {
  // Writes must be sequential in `y`. The natural transcription of ggml's
  // layout emits yy[l], yy[l+32], yy[l+64], yy[l+96] from one loop — four
  // interleaved streams 128 B apart, which Xe cannot merge into wide stores.
  // That alone cost 15x here (2547 -> 39722 Melem/s on a B70, bit-identical),
  // and was what made Q6_K the outlier among the block types. Same arithmetic,
  // reordered so each 32-element run is written in address order.
  float d = (float)b.d;
  for (int idx = 0; idx < 2; ++idx) {
    T *yy = y + 128 * idx;
    const int8_t *sc = b.scales + 8 * idx;
    const uint8_t *ql = b.ql + 64 * idx;
    const uint8_t *qh = b.qh + 32 * idx;
    for (int g = 0; g < 4; ++g) {
      // g selects (which 32 ql bytes, which nibble, which qh bit pair).
      const uint8_t *qlo = ql + 32 * (g & 1);
      const int nsh = 4 * (g >> 1), hsh = 2 * g;
      for (int l = 0; l < 32; ++l) {
        int q = (int)(((qlo[l] >> nsh) & 0xF) | (((qh[l] >> hsh) & 3) << 4)) - 32;
        yy[32 * g + l] = d * sc[2 * g + (l >> 4)] * q;
      }
    }
  }
}

template <typename T>
inline void deq_blk(const BQ8K &b, T *y) {
  for (int j = 0; j < QK_K; ++j) y[j] = b.d * b.qs[j];
}

// Call `f(tag)` with a tag naming the block layout `ggml_dtype` denotes:
// `decltype(tag)::Blk` is the struct, `decltype(tag)::blk` its element count.
template <typename Blk_, int blk_> struct BlkTag {
  using Blk = Blk_;
  static constexpr int blk = blk_;
};

template <typename F> inline int dispatch_blk(uint32_t ggml_dtype, F &&f) {
  switch (ggml_dtype) {
  case G_Q4_0: return f(BlkTag<BQ4_0, QK>{});
  case G_Q4_1: return f(BlkTag<BQ4_1, QK>{});
  case G_Q5_0: return f(BlkTag<BQ5_0, QK>{});
  case G_Q5_1: return f(BlkTag<BQ5_1, QK>{});
  case G_Q8_0: return f(BlkTag<BQ8_0, QK>{});
  case G_Q8_1: return f(BlkTag<BQ8_1, QK>{});
  case G_Q2K: return f(BlkTag<BQ2K, QK_K>{});
  case G_Q3K: return f(BlkTag<BQ3K, QK_K>{});
  case G_Q4K: return f(BlkTag<BQ4K, QK_K>{});
  case G_Q5K: return f(BlkTag<BQ5K, QK_K>{});
  case G_Q6K: return f(BlkTag<BQ6K, QK_K>{});
  case G_Q8K: return f(BlkTag<BQ8K, QK_K>{});
  default: return CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE;
  }
}
