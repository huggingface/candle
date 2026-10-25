// Fused Q8_K prefill GEMM: out[row, col] = lhs[row, k] @ dequant(W[col, k]).
//
// Multi-row lhs (prefill): one thread per output element. The dispatch is
// (1, ceil(n/32), m) workgroups of 32 threads: groupID.z selects the lhs row,
// groupID.y * 32 + dispatchThreadID.x the weight column. Each thread
// dequantizes its column in registers (no shared memory or barriers). W holds
// raw Q8_K blocks (256 elements per 292 bytes: 4-byte f32 scale, then 256 i8
// quants) stored (n, k)-contiguous. k must be a multiple of 256; n need not be
// a multiple of 32 (out-of-range columns early-exit).
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
    uint64_t n = g_Params.n;
    uint64_t k = g_Params.k;
    uint col = groupID.y * 32u + dispatchThreadID.x;
    if (col >= n)
    {
        return;
    }
    uint row = groupID.z;
    uint row_base = uint(row) * uint(k);
    uint kblocks = k / 256u;
    float acc = 0.0;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint off = (col * kblocks + bb) * 292u;
        uint32_t dbits = uint32_t(g_W[off]) | (uint32_t(g_W[off + 1u]) << 8)
            | (uint32_t(g_W[off + 2u]) << 16) | (uint32_t(g_W[off + 3u]) << 24);
        float d = asfloat(dbits);
        uint k0 = bb * 256u;
        for (uint j = 0u; j < 256u; ++j)
        {
            int q = sbyte(g_W[off + 4u + j]);
            acc += g_Lhs[row_base + k0 + j] * (float(q) * d);
        }
    }
    g_Out[uint(row) * uint(n) + col] = acc;
}
