// Fused Q4_K prefill GEMM: out[row, col] = lhs[row, k] @ dequant(W[col, k]).
//
// Multi-row lhs (prefill): one thread per weight column, computing an 8-row
// block of outputs. The dispatch is (1, ceil(n/32), ceil(m/8)) workgroups of
// 32 threads: groupID.z * 8 selects the first lhs row, groupID.y * 32 +
// dispatchThreadID.x the weight column. Each thread dequantizes each column
// element once and reuses it across the 8 rows, so the weight dequantization
// is done once per 8-row block instead of once per (row, col) (8x less
// redundant dequant). The 8 partial sums live in registers (no shared memory,
// no barriers). W holds raw Q4_K blocks (256 elements per 144 bytes: 2 f16
// super scale/min, 12 bytes of 6-bit scale/min pairs, 128 bytes of 4-bit
// quants) stored (n, k)-contiguous. k must be a multiple of 256; n need not be
// a multiple of 32 and m not a multiple of 8 (out-of-range columns/rows are
// guarded per output).
//
// Params: [0]=m, [1]=n, [2]=k.
import dequant_helpers;
import common_fused_gemm;

StructuredBuffer<float> g_Lhs; // (m, k) f32
RWStructuredBuffer<float> g_Out; // (m, n) f32
StructuredBuffer<uint8_t> g_W; // raw Q4_K bytes, (n, k)
layout(push_constant) Params g_Params;

// Q4K/Q5K scale-min decoder (from candle quantized/utils.rs::get_scale_min_k4).
// `off` is the block base offset in g_W; `soff` is the scale-array offset.
uint2 get_scale_min_k4(uint j, uint off, uint soff)
{
    if (j < 4u)
    {
        uint d = g_W[off + soff + j] & 63u;
        uint m = g_W[off + soff + j + 4u] & 63u;
        return uint2(d, m);
    }
    else
    {
        uint d = (g_W[off + soff + j + 4u] & 0xFu) | ((g_W[off + soff + j - 4u] >> 6) << 4);
        uint m = (g_W[off + soff + j + 4u] >> 4) | ((g_W[off + soff + j] >> 6) << 4);
        return uint2(d, m);
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID)
{
    uint64_t m = g_Params.m;
    uint64_t n = g_Params.n;
    uint64_t k = g_Params.k;
    uint col = groupID.y * 32u + dispatchThreadID.x;
    uint row0 = groupID.z * 8u;
    uint kblocks = k / 256u;
    // Eight independent row accumulators (registers, no dynamic indexing).
    float acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
    float acc4 = 0.0, acc5 = 0.0, acc6 = 0.0, acc7 = 0.0;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint off = (col * kblocks + bb) * 144u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
        float mn =
            f16_bits_to_f32(uint16_t(g_W[off + 2u]) | (uint16_t(g_W[off + 3u]) << 8u));
        for (uint s = 0u; s < 8u; ++s)
        {
            uint2 sm = get_scale_min_k4(s, off, 4u);
            float d1 = d * float(sm.x);
            float m1 = mn * float(sm.y);
            uint qbyte0 = off + 16u + 32u * (s / 2u);
            uint k0 = (bb * 8u + s) * 32u;
            if ((s & 1u) == 0u)
            {
                for (uint i = 0u; i < 32u; ++i)
                {
                    float deq = d1 * float(g_W[qbyte0 + i] & 0xFu) - m1;
                    uint64_t gi = uint64_t(k0 + i);
                    if (uint64_t(row0) < m) { acc0 += g_Lhs[uint64_t(row0) * k + gi] * deq; }
                    if (uint64_t(row0 + 1u) < m) { acc1 += g_Lhs[uint64_t(row0 + 1u) * k + gi] * deq; }
                    if (uint64_t(row0 + 2u) < m) { acc2 += g_Lhs[uint64_t(row0 + 2u) * k + gi] * deq; }
                    if (uint64_t(row0 + 3u) < m) { acc3 += g_Lhs[uint64_t(row0 + 3u) * k + gi] * deq; }
                    if (uint64_t(row0 + 4u) < m) { acc4 += g_Lhs[uint64_t(row0 + 4u) * k + gi] * deq; }
                    if (uint64_t(row0 + 5u) < m) { acc5 += g_Lhs[uint64_t(row0 + 5u) * k + gi] * deq; }
                    if (uint64_t(row0 + 6u) < m) { acc6 += g_Lhs[uint64_t(row0 + 6u) * k + gi] * deq; }
                    if (uint64_t(row0 + 7u) < m) { acc7 += g_Lhs[uint64_t(row0 + 7u) * k + gi] * deq; }
                }
            }
            else
            {
                for (uint i = 0u; i < 32u; ++i)
                {
                    float deq = d1 * float((g_W[qbyte0 + i] >> 4u) & 0xFu) - m1;
                    uint64_t gi = uint64_t(k0 + i);
                    if (uint64_t(row0) < m) { acc0 += g_Lhs[uint64_t(row0) * k + gi] * deq; }
                    if (uint64_t(row0 + 1u) < m) { acc1 += g_Lhs[uint64_t(row0 + 1u) * k + gi] * deq; }
                    if (uint64_t(row0 + 2u) < m) { acc2 += g_Lhs[uint64_t(row0 + 2u) * k + gi] * deq; }
                    if (uint64_t(row0 + 3u) < m) { acc3 += g_Lhs[uint64_t(row0 + 3u) * k + gi] * deq; }
                    if (uint64_t(row0 + 4u) < m) { acc4 += g_Lhs[uint64_t(row0 + 4u) * k + gi] * deq; }
                    if (uint64_t(row0 + 5u) < m) { acc5 += g_Lhs[uint64_t(row0 + 5u) * k + gi] * deq; }
                    if (uint64_t(row0 + 6u) < m) { acc6 += g_Lhs[uint64_t(row0 + 6u) * k + gi] * deq; }
                    if (uint64_t(row0 + 7u) < m) { acc7 += g_Lhs[uint64_t(row0 + 7u) * k + gi] * deq; }
                }
            }
        }
    }
    if (col < n)
    {
        uint64_t ob = uint64_t(col);
        if (uint64_t(row0) < m) { g_Out[uint64_t(row0) * n + ob] = acc0; }
        if (uint64_t(row0 + 1u) < m) { g_Out[uint64_t(row0 + 1u) * n + ob] = acc1; }
        if (uint64_t(row0 + 2u) < m) { g_Out[uint64_t(row0 + 2u) * n + ob] = acc2; }
        if (uint64_t(row0 + 3u) < m) { g_Out[uint64_t(row0 + 3u) * n + ob] = acc3; }
        if (uint64_t(row0 + 4u) < m) { g_Out[uint64_t(row0 + 4u) * n + ob] = acc4; }
        if (uint64_t(row0 + 5u) < m) { g_Out[uint64_t(row0 + 5u) * n + ob] = acc5; }
        if (uint64_t(row0 + 6u) < m) { g_Out[uint64_t(row0 + 6u) * n + ob] = acc6; }
        if (uint64_t(row0 + 7u) < m) { g_Out[uint64_t(row0 + 7u) * n + ob] = acc7; }
    }
}