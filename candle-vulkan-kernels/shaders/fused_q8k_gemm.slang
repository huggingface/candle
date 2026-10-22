// Fused Q8_K decode GEMM: out[n] = lhs[k] @ dequant(W[n, k]).
//
// Single-row lhs (decode): each thread owns one output column and
// dequantizes its Q8_K column in registers while accumulating the dot
// product, so no shared memory or barriers are needed and every thread
// does useful work. W holds raw Q8_K blocks (256 elements per 292 bytes: 4-byte f32 scale, then 256 i8 quants) stored
// (n, k)-contiguous. m must be 1 and k a multiple of 256; n need not be a
// multiple of 32 (out-of-range columns early-exit).
//
// Params: [0]=m (must be 1), [1]=n, [2]=k.
import dequant_helpers;

StructuredBuffer<float> g_Lhs; // (k,) f32
RWStructuredBuffer<float> g_Out; // (n,) f32
StructuredBuffer<uint8_t> g_W; // raw Q8_K bytes, (n, k)
StructuredBuffer<uint64_t> g_Params;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID)
{
    uint64_t n = g_Params[1];
    uint64_t k = g_Params[2];
    uint col = groupID.y * 32u + dispatchThreadID.x;
    if (col >= n)
    {
        return;
    }
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
            acc += g_Lhs[k0 + j] * (float(q) * d);
        }
    }
    g_Out[col] = acc;
}
