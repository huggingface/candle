// dequantize _q6k

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


    uint off = b * 210u;
    float d = f16_bits_to_f32(uint16_t(g_Src[off + 208u]) | (uint16_t(g_Src[off + 209u]) << 8));
    for (uint e = 0u; e < 256u; ++e)
    {
        uint idx = e / 128u;
        uint l = e % 32u;
        uint group = (e % 128u) / 32u;
        uint is = l / 16u;
        uint sc_idx = 8u * idx + is + 2u * group;
        uint ql_byte = off + 64u * idx + l + 32u * (group % 2u);
        uint qh_byte = off + 128u + 32u * idx + l;
        uint qlv = g_Src[ql_byte];
        uint qhv = g_Src[qh_byte];
        uint low4 = (group < 2u) ? (qlv & 0xFu) : (qlv >> 4);
        uint high2;
        if (group == 0u)
        {
            high2 = qhv & 3u;
        }
        else if (group == 1u)
        {
            high2 = (qhv >> 2) & 3u;
        }
        else if (group == 2u)
        {
            high2 = (qhv >> 4) & 3u;
        }
        else
        {
            high2 = (qhv >> 6) & 3u;
        }
        int q6 = int(low4 | (high2 << 4)) - 32;
        int scv = sbyte(g_Src[off + 192u + sc_idx]);
        g_Dst[b * 256u + e] = d * float(scv) * float(q6);
    }

}
