// Fused Q5_K decode GEMM: out[n] = lhs[k] @ dequant(W[n, k]).
//
// Single-row lhs (decode): each thread owns one output column and
// dequantizes its Q5_K column in registers while accumulating the dot
// product, so no shared memory or barriers are needed and every thread
// does useful work. W holds raw Q5_K blocks (256 elements per 176 bytes: 2 f16 super scale/min, 12 bytes of 6-bit scale/min pairs, 32 bytes of high bits, 128 bytes of 4-bit quants) stored
// (n, k)-contiguous. m must be 1 and k a multiple of 256; n need not be a
// multiple of 32 (out-of-range columns early-exit).
//
// Params: [0]=m (must be 1), [1]=n, [2]=k.
StructuredBuffer<float> g_Lhs; // (k,) f32
RWStructuredBuffer<float> g_Out; // (n,) f32
StructuredBuffer<uint8_t> g_W; // raw Q5_K bytes, (n, k)
StructuredBuffer<uint64_t> g_Params;

// Reinterpret an IEEE-754 half (16 bits) as a float (half crate).
float f16_bits_to_f32(uint16_t h)
{
    uint sign = (h >> 15) & 1u;
    uint exp = (h >> 10) & 0x1Fu;
    uint mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0u)
    {
        if (mant == 0u)
        {
            f = sign << 31; // signed zero
        }
        else
        {
            uint m = mant;
            uint shift = 0u;
            while ((m & 0x400u) == 0u)
            {
                m <<= 1;
                shift += 1;
            }
            m &= 0x3FFu;
            f = (sign << 31) | ((113u - shift) << 23) | (m << 13);
        }
    }
    else if (exp == 31u)
    {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    }
    else
    {
        f = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    return asfloat(f);
}
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
        uint off = (col * kblocks + bb) * 176u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
        float mn =
            f16_bits_to_f32(uint16_t(g_W[off + 2u]) | (uint16_t(g_W[off + 3u]) << 8u));
        for (uint s = 0u; s < 8u; ++s)
        {
            uint2 sm = get_scale_min_k4(s, off, 4u);
            float d1 = d * float(sm.x);
            float m1 = mn * float(sm.y);
            uint ql0 = off + 48u + 32u * (s / 2u);
            uint k0 = (bb * 8u + s) * 32u;
            uint bit = 1u << s;
            if ((s & 1u) == 0u)
            {
                for (uint i = 0u; i < 32u; ++i)
                {
                    uint qv = g_W[ql0 + i] & 0xFu;
                    uint to_add = (g_W[off + 16u + i] & bit) != 0u ? 16u : 0u;
                    acc += g_Lhs[k0 + i] * (d1 * float(qv + to_add) - m1);
                }
            }
            else
            {
                for (uint i = 0u; i < 32u; ++i)
                {
                    uint qv = (g_W[ql0 + i] >> 4u) & 0xFu;
                    uint to_add = (g_W[off + 16u + i] & bit) != 0u ? 16u : 0u;
                    acc += g_Lhs[k0 + i] * (d1 * float(qv + to_add) - m1);
                }
            }
        }
    }
    g_Out[col] = acc;
}
