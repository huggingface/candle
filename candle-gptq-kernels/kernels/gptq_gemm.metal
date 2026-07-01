// Fused dequantize + GEMM kernel for GPTQ (AutoGPTQ/GPTQModel) "old" layout, Metal port.
//
// Computes Y[M,N] = X[M,K] @ dequant(qweight,qzeros,scales,g_idx)[K,N] using a
// threadgroup-memory-tiled GEMM, dequantizing each weight element on the fly while it is staged
// into threadgroup memory. Accumulation is scalar FP32. This is a straight port of
// `kernels/gptq_gemm.cu` (CUDA), with TILE reduced to 16 so 16x16=256 threads fit comfortably in
// a Metal threadgroup.
#include <metal_stdlib>
using namespace metal;

#define TILE 16

struct GptqParams {
    int M;
    int K;
    int N;
    int bits;
    int pack_factor;
    int n_groups_out; // N / pack_factor
};

kernel void gptq_gemm_f32(
    device const float *x         [[buffer(0)]], // [M, K]
    device const int   *qweight   [[buffer(1)]], // [K / pack_factor, N]
    device const int   *qzeros    [[buffer(2)]], // [n_groups, N / pack_factor]
    device const float *scales    [[buffer(3)]], // [n_groups, N]
    device const int   *g_idx     [[buffer(4)]], // [K]
    device float       *y         [[buffer(5)]], // [M, N]
    constant GptqParams &p        [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    const int M = p.M;
    const int K = p.K;
    const int N = p.N;
    const int bits = p.bits;
    const int pack_factor = p.pack_factor;
    const int n_groups_out = p.n_groups_out;
    const int mask = (1 << bits) - 1;

    const int row = int(gid.y) * TILE + int(tid.y); // index into M
    const int col = int(gid.x) * TILE + int(tid.x); // index into N

    float acc = 0.0f;
    const int n_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int a_col = t * TILE + int(tid.x);
        As[tid.y][tid.x] = (row < M && a_col < K) ? x[row * K + a_col] : 0.0f;

        const int k = t * TILE + int(tid.y);
        if (k < K && col < N) {
            const int g = g_idx[k];
            const int w_word = qweight[(k / pack_factor) * N + col];
            const int shift_q = (k % pack_factor) * bits;
            const int q = (w_word >> shift_q) & mask;

            const int z_word = qzeros[g * n_groups_out + col / pack_factor];
            const int shift_z = (col % pack_factor) * bits;
            const int z = ((z_word >> shift_z) & mask) + 1;

            const float s = scales[g * N + col];
            Bs[tid.y][tid.x] = (float)(q - z) * s;
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
