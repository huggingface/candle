// Fused Q8_K prefill GEMM: out[row, col] = lhs[row, k] @ dequant(W[col, k]).
//
// Multi-row lhs (prefill): one thread per weight column, computing an 8-row
// block of outputs. The dispatch is (1, ceil(n/32), ceil(m/8)) workgroups of
// 32 threads: groupID.z * 8 selects the first lhs row. Each thread dequantizes
// each column element once and reuses it across the 8 rows (8x less redundant
// dequant than one-thread-per-(row,col)); the 8 partial sums live in registers.
// W holds raw Q8_K blocks (256 elements per 292 bytes: 4-byte f32 scale, 256
// i8 quants) stored (n, k)-contiguous. k must be a multiple of 256.
//
// Params: [0]=m, [1]=n, [2]=k.
import dequant_helpers;
import common_fused_gemm;

StructuredBuffer<float> g_Lhs; // (m, k) f32
RWStructuredBuffer<float> g_Out; // (m, n) f32
StructuredBuffer<uint8_t> g_W; // raw Q8_K bytes, (n, k)
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
        uint off = (col * kblocks + bb) * 292u;
        uint32_t dbits = uint32_t(g_W[off]) | (uint32_t(g_W[off + 1u]) << 8)
            | (uint32_t(g_W[off + 2u]) << 16) | (uint32_t(g_W[off + 3u]) << 24);
        float d = asfloat(dbits);
        uint k0 = bb * 256u;
        for (uint j = 0u; j < 256u; ++j)
        {
            float deq = float(sbyte(g_W[off + 4u + j])) * d;
            uint64_t gi = uint64_t(k0 + j);
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