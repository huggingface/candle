// Fused Q8_0 decode GEMM via DP4A (V_DOT4C_I32_I8):
// out[n] = lhs[k] @ dequant(W[n, k]).
//
// Single-row lhs (decode): the activation has already been quantized to i8 by
// `quantize_lhs_i8` (see that shader), so each output column runs the integer
// dot product over packed 4x8-bit quants instead of f32 FMAs. W holds raw
// Q8_0 blocks (32 elements per 34 bytes: 2-byte f16 scale, then 32 i8 quants)
// stored (n, k)-contiguous. m must be 1 and k a multiple of 32.
//
// Params: [0]=m (must be 1), [1]=n, [2]=k.
import dequant_helpers;
import common_fused_gemm;

StructuredBuffer<uint8_t> g_LhsQ; // (k,) i8 activation (2's-complement bytes)
StructuredBuffer<uint8_t> g_W; // raw Q8_0 bytes, (n, k)
StructuredBuffer<float> g_Scale; // (1,) f32 dequant scale (max/127)
RWStructuredBuffer<float> g_Out; // (n,) f32
layout(push_constant) Params g_Params;

// Pack four signed i8 into a u32 (little-endian bytes) for `dot4add_i8packed`.
uint pack4i8(int v0, int v1, int v2, int v3)
{
    return (uint(v0) & 0xFFu)
        | ((uint(v1) & 0xFFu) << 8u)
        | ((uint(v2) & 0xFFu) << 16u)
        | ((uint(v3) & 0xFFu) << 24u);
}

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
    uint kblocks = uint(k / 32u);
    float s = g_Scale[0];
    // Per-block i32 dot (each Q8_0 block has its own f16 scale `d`), summed
    // in f32 across blocks.
    float facc = 0.0f;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint off = (col * kblocks + bb) * 34u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
        int bacc = 0;
        for (uint j = 0u; j < 32u; j += 4u)
        {
            uint qp = pack4i8(
                sbyte(g_W[off + 2u + j]),
                sbyte(g_W[off + 2u + j + 1u]),
                sbyte(g_W[off + 2u + j + 2u]),
                sbyte(g_W[off + 2u + j + 3u]));
            uint lp = pack4i8(
                sbyte(g_LhsQ[bb * 32u + j]),
                sbyte(g_LhsQ[bb * 32u + j + 1u]),
                sbyte(g_LhsQ[bb * 32u + j + 2u]),
                sbyte(g_LhsQ[bb * 32u + j + 3u]));
            bacc = dot4add_i8packed(lp, qp, bacc);
        }
        facc += d * float(bacc);
    }
    g_Out[col] = s * facc;
}