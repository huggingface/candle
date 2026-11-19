// Fused Q6_K prefill GEMM: out[row, col] = lhs[row, k] @ dequant(W[col, k]).
//
// Multi-row lhs (prefill): one thread per weight column, computing an 8-row
// block of outputs. The dispatch is (1, ceil(n/32), ceil(m/8)) workgroups of
// 32 threads: groupID.z * 8 selects the first lhs row. Each thread dequantizes
// each column element once and reuses it across the 8 rows (8x less redundant
// dequant than one-thread-per-(row,col)); the 8 partial sums live in registers.
// W holds raw Q6_K blocks (256 elements per 210 bytes: 128 bytes of 4-bit low
// quants at [0..128), 64 bytes of 2-bit high quants at [128..192), 16 i8
// scales (one per 16 elements) at [192..208), and an f16 super-scale d at
// [208..210)) stored (n, k)-contiguous. k must be a multiple of 256.
//
// Decode layout: see fused_q6k_gemm.slang.
//
// Params: [0]=m, [1]=n, [2]=k.
import dequant_helpers;
import common_fused_gemm;

StructuredBuffer<float> g_Lhs; // (m, k) f32
RWStructuredBuffer<float> g_Out; // (m, n) f32
StructuredBuffer<uint8_t> g_W; // raw Q6_K bytes, (n, k)
layout(push_constant) Params g_Params;

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
    float acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
    float acc4 = 0.0, acc5 = 0.0, acc6 = 0.0, acc7 = 0.0;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint off = (col * kblocks + bb) * 210u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off + 208u]) | (uint16_t(g_W[off + 209u]) << 8u));
        for (uint idx = 0u; idx < 2u; ++idx)
        {
            for (uint l = 0u; l < 32u; ++l)
            {
                uint is = l / 16u;
                uint ql1 = g_W[off + 64u * idx + l];
                uint ql2 = g_W[off + 64u * idx + l + 32u];
                uint qh = g_W[off + 128u + 32u * idx + l];
                float deq1 = d * (float(int((ql1 & 0xFu) | ((qh & 3u) << 4)) - 32));
                float deq2 = d * (float(int((ql2 & 0xFu) | (((qh >> 2u) & 3u) << 4)) - 32));
                float deq3 = d * (float(int((ql1 >> 4u) | (((qh >> 4u) & 3u) << 4)) - 32));
                float deq4 = d * (float(int((ql2 >> 4u) | (((qh >> 6u) & 3u) << 4)) - 32));
                uint sc = off + 192u + 8u * idx + is;
                deq1 *= float(sbyte(g_W[sc]));
                deq2 *= float(sbyte(g_W[sc + 2u]));
                deq3 *= float(sbyte(g_W[sc + 4u]));
                deq4 *= float(sbyte(g_W[sc + 6u]));
                uint64_t gi = uint64_t(bb * 256u + 128u * idx + l);
                if (uint64_t(row0) < m) { acc0 += g_Lhs[uint64_t(row0) * k + gi] * deq1; }
                if (uint64_t(row0 + 1u) < m) { acc1 += g_Lhs[uint64_t(row0 + 1u) * k + gi] * deq1; }
                if (uint64_t(row0 + 2u) < m) { acc2 += g_Lhs[uint64_t(row0 + 2u) * k + gi] * deq1; }
                if (uint64_t(row0 + 3u) < m) { acc3 += g_Lhs[uint64_t(row0 + 3u) * k + gi] * deq1; }
                if (uint64_t(row0 + 4u) < m) { acc4 += g_Lhs[uint64_t(row0 + 4u) * k + gi] * deq1; }
                if (uint64_t(row0 + 5u) < m) { acc5 += g_Lhs[uint64_t(row0 + 5u) * k + gi] * deq1; }
                if (uint64_t(row0 + 6u) < m) { acc6 += g_Lhs[uint64_t(row0 + 6u) * k + gi] * deq1; }
                if (uint64_t(row0 + 7u) < m) { acc7 += g_Lhs[uint64_t(row0 + 7u) * k + gi] * deq1; }
                gi += 32u;
                if (uint64_t(row0) < m) { acc0 += g_Lhs[uint64_t(row0) * k + gi] * deq2; }
                if (uint64_t(row0 + 1u) < m) { acc1 += g_Lhs[uint64_t(row0 + 1u) * k + gi] * deq2; }
                if (uint64_t(row0 + 2u) < m) { acc2 += g_Lhs[uint64_t(row0 + 2u) * k + gi] * deq2; }
                if (uint64_t(row0 + 3u) < m) { acc3 += g_Lhs[uint64_t(row0 + 3u) * k + gi] * deq2; }
                if (uint64_t(row0 + 4u) < m) { acc4 += g_Lhs[uint64_t(row0 + 4u) * k + gi] * deq2; }
                if (uint64_t(row0 + 5u) < m) { acc5 += g_Lhs[uint64_t(row0 + 5u) * k + gi] * deq2; }
                if (uint64_t(row0 + 6u) < m) { acc6 += g_Lhs[uint64_t(row0 + 6u) * k + gi] * deq2; }
                if (uint64_t(row0 + 7u) < m) { acc7 += g_Lhs[uint64_t(row0 + 7u) * k + gi] * deq2; }
                gi += 32u;
                if (uint64_t(row0) < m) { acc0 += g_Lhs[uint64_t(row0) * k + gi] * deq3; }
                if (uint64_t(row0 + 1u) < m) { acc1 += g_Lhs[uint64_t(row0 + 1u) * k + gi] * deq3; }
                if (uint64_t(row0 + 2u) < m) { acc2 += g_Lhs[uint64_t(row0 + 2u) * k + gi] * deq3; }
                if (uint64_t(row0 + 3u) < m) { acc3 += g_Lhs[uint64_t(row0 + 3u) * k + gi] * deq3; }
                if (uint64_t(row0 + 4u) < m) { acc4 += g_Lhs[uint64_t(row0 + 4u) * k + gi] * deq3; }
                if (uint64_t(row0 + 5u) < m) { acc5 += g_Lhs[uint64_t(row0 + 5u) * k + gi] * deq3; }
                if (uint64_t(row0 + 6u) < m) { acc6 += g_Lhs[uint64_t(row0 + 6u) * k + gi] * deq3; }
                if (uint64_t(row0 + 7u) < m) { acc7 += g_Lhs[uint64_t(row0 + 7u) * k + gi] * deq3; }
                gi += 32u;
                if (uint64_t(row0) < m) { acc0 += g_Lhs[uint64_t(row0) * k + gi] * deq4; }
                if (uint64_t(row0 + 1u) < m) { acc1 += g_Lhs[uint64_t(row0 + 1u) * k + gi] * deq4; }
                if (uint64_t(row0 + 2u) < m) { acc2 += g_Lhs[uint64_t(row0 + 2u) * k + gi] * deq4; }
                if (uint64_t(row0 + 3u) < m) { acc3 += g_Lhs[uint64_t(row0 + 3u) * k + gi] * deq4; }
                if (uint64_t(row0 + 4u) < m) { acc4 += g_Lhs[uint64_t(row0 + 4u) * k + gi] * deq4; }
                if (uint64_t(row0 + 5u) < m) { acc5 += g_Lhs[uint64_t(row0 + 5u) * k + gi] * deq4; }
                if (uint64_t(row0 + 6u) < m) { acc6 += g_Lhs[uint64_t(row0 + 6u) * k + gi] * deq4; }
                if (uint64_t(row0 + 7u) < m) { acc7 += g_Lhs[uint64_t(row0 + 7u) * k + gi] * deq4; }
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