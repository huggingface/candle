// Fused Q8_0 decode GEMM: out[n] = lhs[k] @ dequant(W[n, k]).
//
// Single-row lhs (decode): each thread owns one output column and
// dequantizes its Q8_0 column in registers while accumulating the dot
// product, so no shared memory or barriers are needed and every thread
// does useful work. W holds raw Q8_0 blocks (32 elements per 34 bytes: 2-byte f16 scale, then 32 i8 quants) stored
// (n, k)-contiguous. m must be 1 and k a multiple of 32; n need not be a
// multiple of 32 (out-of-range columns early-exit).
//
// Params: [0]=m (must be 1), [1]=n, [2]=k.
import dequant_helpers;
import common_fused_gemm;

StructuredBuffer<float> g_Lhs; // (k,) f32
RWStructuredBuffer<float> g_Out; // (n,) f32
StructuredBuffer<uint8_t> g_W; // raw Q8_0 bytes, (n, k)
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
    uint kblocks = k / 32u;
    float acc = 0.0;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint off = (col * kblocks + bb) * 34u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
        for (uint j = 0u; j < 32u; ++j)
        {
            int q = sbyte(g_W[off + 2u + j]);
            acc += g_Lhs[bb * 32u + j] * (float(q) * d);
        }
    }
    g_Out[col] = acc;
}
