// Fused dequantize + GEMM kernel for AWQ (AutoAWQ "GEMM" layout), 4-bit only, Metal port.
//
// Straight port of `kernels/awq_gemm.cu`. AWQ packs 4-bit values along the *output*-feature axis
// with a fixed nibble permutation (inherited from the original llm-awq CUDA kernel) and uses zero
// points as-is (no `+1` offset). TILE is reduced to 16 so 16x16=256 threads fit comfortably in a
// Metal threadgroup.
#include <metal_stdlib>
using namespace metal;

#define TILE 16
#define AWQ_BITS 4
#define AWQ_PACK_FACTOR 8

struct AwqParams {
    int M;
    int K;
    int N;
    int group_size;
    int n_packed_out; // N / pack_factor
};

// Inverse of AutoAWQ's packing `order_map = [0, 2, 4, 6, 1, 3, 5, 7]`: in-word position `i`
// belongs to output column `col_group * 8 + AWQ_ORDER[i]`.
constant int AWQ_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};

kernel void awq_gemm_f32(
    device const float *x        [[buffer(0)]], // [M, K]
    device const int   *qweight  [[buffer(1)]], // [K, N / pack_factor]
    device const int   *qzeros   [[buffer(2)]], // [n_groups, N / pack_factor]
    device const float *scales   [[buffer(3)]], // [n_groups, N]
    device float       *y        [[buffer(4)]], // [M, N]
    constant AwqParams &p        [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    const int M = p.M;
    const int K = p.K;
    const int N = p.N;
    const int group_size = p.group_size;
    const int n_packed_out = p.n_packed_out;
    const int mask = (1 << AWQ_BITS) - 1;

    const int row = int(gid.y) * TILE + int(tid.y); // index into M
    const int col = int(gid.x) * TILE + int(tid.x); // index into N

    float acc = 0.0f;
    const int n_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const int a_col = t * TILE + int(tid.x);
        As[tid.y][tid.x] = (row < M && a_col < K) ? x[row * K + a_col] : 0.0f;

        const int k = t * TILE + int(tid.y);
        if (k < K && col < N) {
            const int g = k / group_size;
            const int col_group = col / AWQ_PACK_FACTOR;
            const int j = col % AWQ_PACK_FACTOR;
            const int shift = AWQ_ORDER[j] * AWQ_BITS;

            const int w_word = qweight[k * n_packed_out + col_group];
            const int q = (w_word >> shift) & mask;

            const int z_word = qzeros[g * n_packed_out + col_group];
            const int z = (z_word >> shift) & mask;

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
