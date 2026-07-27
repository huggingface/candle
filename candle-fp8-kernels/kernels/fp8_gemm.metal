// Fused dequantize + GEMM kernel for block-wise FP8 (E4M3) checkpoints (DeepSeek-V3 style),
// Metal port of `kernels/fp8_gemm.cu`.
//
// Computes Y[M,N] = X[M,K] @ W^T, where W is stored as `[N, K]` F8E4M3 plus a per-block f32 scale
// `scale[N / block_size, K / block_size]` (`W[n,k] = fp8[n,k] * scale[n/bs, k/bs]`). Metal has no
// native fp8 type, so each E4M3 byte is decoded to f32 in the shader. TILE is reduced to 16 so
// 16x16=256 threads fit comfortably in a Metal threadgroup.
#include <metal_stdlib>
using namespace metal;

#define TILE 16

struct Fp8Params {
    int M;
    int K;
    int N;
    int block_size;
    int scale_cols; // ceil(K / block_size)
};

// Decode an OCP/NVIDIA E4M3 byte (1 sign, 4 exp bias-7, 3 mantissa) to f32. Matches the `float8`
// crate's `F8E4M3 -> f32` for finite values; quantized weights never carry the NaN encoding
// (exp=15, mant=7), so it is not special-cased.
static inline float decode_e4m3(uchar b) {
    uint sign = (b >> 7) & 0x1u;
    uint exp = (b >> 3) & 0xFu;
    uint mant = b & 0x7u;
    float val;
    if (exp == 0u) {
        // Subnormal: 2^(1 - bias) * mant/8, bias = 7.
        val = (float(mant) / 8.0f) * exp2(-6.0f);
    } else {
        val = (1.0f + float(mant) / 8.0f) * exp2(float(int(exp) - 7));
    }
    return sign != 0u ? -val : val;
}

kernel void fp8_block_gemm_f32(
    device const float *x      [[buffer(0)]], // [M, K]
    device const uchar *w      [[buffer(1)]], // [N, K] E4M3 bytes
    device const float *scale  [[buffer(2)]], // [N / block_size, K / block_size]
    device float       *y      [[buffer(3)]], // [M, N]
    constant Fp8Params &p      [[buffer(4)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    const int M = p.M;
    const int K = p.K;
    const int N = p.N;
    const int block_size = p.block_size;
    const int scale_cols = p.scale_cols;

    const int row = int(gid.y) * TILE + int(tid.y); // index into M
    const int col = int(gid.x) * TILE + int(tid.x); // index into N

    float acc = 0.0f;
    const int n_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int a_col = t * TILE + int(tid.x);
        As[tid.y][tid.x] = (row < M && a_col < K) ? x[row * K + a_col] : 0.0f;

        const int k = t * TILE + int(tid.y);
        if (k < K && col < N) {
            const float w_val = decode_e4m3(w[col * K + k]);
            const float s = scale[(col / block_size) * scale_cols + k / block_size];
            Bs[tid.y][tid.x] = w_val * s;
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < TILE; ++i) {
            acc += As[tid.y][i] * Bs[i][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        y[row * N + col] = acc;
    }
}
