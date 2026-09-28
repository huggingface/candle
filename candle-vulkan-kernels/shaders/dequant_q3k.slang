// dequantize _q3k

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


    uint off = b * 110u;
    float d = f16_bits_to_f32(uint16_t(g_Src[off + 108u]) | (uint16_t(g_Src[off + 109u]) << 8));
    // Reconstruct the 16 scale bytes from the 12-byte packed scales (Q3K layout).
    uint KMASK1 = 0x03030303u;
    uint KMASK2 = 0x0F0F0F0Fu;
    uint a0 = uint32_t(g_Src[off + 96u]) | (uint32_t(g_Src[off + 97u]) << 8)
        | (uint32_t(g_Src[off + 98u]) << 16) | (uint32_t(g_Src[off + 99u]) << 24);
    uint a1 = uint32_t(g_Src[off + 100u]) | (uint32_t(g_Src[off + 101u]) << 8)
        | (uint32_t(g_Src[off + 102u]) << 16) | (uint32_t(g_Src[off + 103u]) << 24);
    uint a2 = uint32_t(g_Src[off + 104u]) | (uint32_t(g_Src[off + 105u]) << 8)
        | (uint32_t(g_Src[off + 106u]) << 16) | (uint32_t(g_Src[off + 107u]) << 24);
    uint tmp = a2;
    a2 = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    uint a3 = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    a0 = (a0 & KMASK2) | (((tmp) & KMASK1) << 4);
    a1 = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    for (uint e = 0u; e < 256u; ++e)
    {
        uint yc = e / 128u;
        uint sl = e % 128u;
        uint s = sl / 32u;
        uint w32 = sl % 32u;
        uint scidx = w32 / 16u;
        uint ii = w32 % 16u;
        uint is = yc * 8u + s * 2u + scidx;
        uint shift = 2u * s;
        uint m = 1u << (yc * 4u + s);
        uint qbyte = off + 32u + yc * 32u + ii + 16u * scidx;
        uint hbyte = ii + 16u * scidx;
        uint v = (g_Src[qbyte] >> shift) & 3u;
        uint correction = (g_Src[off + hbyte] & m) == 0u ? 4u : 0u;
        uint word = (is / 4u) == 0u ? a0 : (is / 4u) == 1u ? a1 : (is / 4u) == 2u ? a2 : a3;
        int scale_val = sbyte((word >> (8u * (is % 4u))) & 0xFFu);
        g_Dst[b * 256u + e] = d * float(scale_val - 32) * float(int(v) - int(correction));
    }

}
