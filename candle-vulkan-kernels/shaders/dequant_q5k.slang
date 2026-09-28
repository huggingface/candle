// dequantize _q5k

StructuredBuffer<uint8_t> g_Src;
RWStructuredBuffer<float> g_Dst;
StructuredBuffer<float> g_Params; // [0]=num_blocks


// Reinterpret an IEEE-754 half (16 bits) as a float.
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
            f = sign << 31;
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
    return (u & 0x80u) ? (int(u) - 256) : int(u);
}
// Q4K/Q5K scale-min decoder (from candle quantized/utils.rs::get_scale_min_k4).
// `off` is the block base offset in g_Src; `soff` is the scale-array offset.
uint2 get_scale_min_k4(uint j, uint off, uint soff)
{
    if (j < 4u)
    {
        uint d = g_Src[off + soff + j] & 63u;
        uint m = g_Src[off + soff + j + 4u] & 63u;
        return uint2(d, m);
    }
    else
    {
        uint d = (g_Src[off + soff + j + 4u] & 0xFu) | ((g_Src[off + soff + j - 4u] >> 6) << 4);
        uint m = (g_Src[off + soff + j + 4u] >> 4) | ((g_Src[off + soff + j] >> 6) << 4);
        return uint2(d, m);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint b = dispatchThreadID.x;
    if (b >= uint(g_Params[0]))
    {
        return;
    }


    uint off = b * 176u;
    float d = f16_bits_to_f32(uint16_t(g_Src[off]) | (uint16_t(g_Src[off + 1]) << 8));
    float mn = f16_bits_to_f32(uint16_t(g_Src[off + 2]) | (uint16_t(g_Src[off + 3]) << 8));
    for (uint e = 0u; e < 256u; ++e)
    {
        uint jstep = e / 64u;
        uint within = e % 64u;
        bool first = within < 32u;
        uint i = within % 32u;
        uint ql_byte = off + 48u + 32u * jstep + i;
        uint qh_byte = off + 16u + i;
        uint bit = 1u << (2u * jstep + (first ? 0u : 1u));
        uint to_add = (g_Src[qh_byte] & bit) != 0u ? 16u : 0u;
        uint qv = first ? (g_Src[ql_byte] & 0xFu) : (g_Src[ql_byte] >> 4);
        uint2 sm = get_scale_min_k4(2u * jstep + (first ? 0u : 1u), off, 4u);
        float d1 = d * float(sm.x);
        float m1 = mn * float(sm.y);
        g_Dst[b * 256u + e] = d1 * float(qv + to_add) - m1;
    }

}
