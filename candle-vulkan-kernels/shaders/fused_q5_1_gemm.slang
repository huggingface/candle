// Fused Q5_1 decode GEMM: out[n] = lhs[k] @ dequant(W[n, k]).
//
// Single-row lhs (decode): each thread owns one output column and
// dequantizes its Q5_1 column in registers while accumulating the dot
// product, so no shared memory or barriers are needed and every thread
// does useful work. W holds raw Q5_1 blocks (32 elements per 24 bytes: 2 f16 scale/min, 4 bytes of high bits, 16 bytes of 4-bit quants) stored
// (n, k)-contiguous. m must be 1 and k a multiple of 32; n need not be a
// multiple of 32 (out-of-range columns early-exit).
//
// Params: [0]=m (must be 1), [1]=n, [2]=k.
import dequant_helpers;

StructuredBuffer<float> g_Lhs; // (k,) f32
RWStructuredBuffer<float> g_Out; // (n,) f32
StructuredBuffer<uint8_t> g_W; // raw Q5_1 bytes, (n, k)
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
    uint kblocks = k / 32u;
    float acc = 0.0;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint off = (col * kblocks + bb) * 24u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
        float m =
            f16_bits_to_f32(uint16_t(g_W[off + 2u]) | (uint16_t(g_W[off + 3u]) << 8u));
        uint32_t qh = uint32_t(g_W[off + 4u]) | (uint32_t(g_W[off + 5u]) << 8)
            | (uint32_t(g_W[off + 6u]) << 16) | (uint32_t(g_W[off + 7u]) << 24);
        for (uint j = 0u; j < 16u; ++j)
        {
            uint q = g_W[off + 8u + j];
            uint xh0 = ((qh >> j) << 4u) & 0x10u;
            uint xh1 = (qh >> (j + 12u)) & 0x10u;
            float x0 = float(int((q & 0x0Fu) | xh0));
            float x1 = float(int(((q >> 4u) & 0x0Fu) | xh1));
            acc += g_Lhs[bb * 32u + j] * (x0 * d + m);
            acc += g_Lhs[bb * 32u + 16u + j] * (x1 * d + m);
        }
    }
    g_Out[col] = acc;
}
