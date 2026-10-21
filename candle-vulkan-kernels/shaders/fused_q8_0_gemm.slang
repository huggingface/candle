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
StructuredBuffer<float> g_Lhs; // (k,) f32
RWStructuredBuffer<float> g_Out; // (n,) f32
StructuredBuffer<uint8_t> g_W; // raw Q8_0 bytes, (n, k)
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
// Reinterpret a byte as a signed int8.
int sbyte(uint u)
{
    return (u & 0x80u) != 0u ? (int(u) - 256) : int(u);
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
