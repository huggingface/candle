// Fused Q4_0 decode GEMM: out[n] = lhs[k] @ dequant(W[n, k]).
//
// Single-row lhs (decode): each thread owns one output column and
// dequantizes its Q4_0 column in registers while accumulating the dot
// product, so no shared memory or barriers are needed and every thread
// does useful work. W holds raw Q4_0 blocks (32 elements per 18 bytes:
// 2-byte f16 scale, then 16 bytes holding 32 4-bit quants — byte j holds
// element j in its low nibble and element 16+j in its high nibble (the
// pre-2024 llama.cpp Q4_0 layout used by candle's BlockQ4_0) — stored
// (n, k)-contiguous. m must be 1 and k a multiple of 32; n need not be a
// multiple of 32 (out-of-range columns early-exit).
//
// Params: [0]=m (must be 1), [1]=n, [2]=k.
import dequant_helpers;
import common_fused_gemm;

StructuredBuffer<float> g_Lhs; // (k,) f32
RWStructuredBuffer<float> g_Out; // (n,) f32
StructuredBuffer<uint8_t> g_W; // raw Q4_0 bytes, (n, k)
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
        uint off = (col * kblocks + bb) * 18u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
        uint k0 = bb * 32u;
        for (uint j = 0u; j < 16u; ++j)
        {
            uint q = g_W[off + 2u + j];
            float x0 = float(int(q & 0x0Fu)) - 8.0;
            float x1 = float(int((q >> 4u) & 0x0Fu)) - 8.0;
            acc += g_Lhs[k0 + j] * (x0 * d);
            acc += g_Lhs[k0 + 16u + j] * (x1 * d);
        }
    }
    g_Out[col] = acc;
}
