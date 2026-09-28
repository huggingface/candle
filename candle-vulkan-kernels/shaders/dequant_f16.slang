// dequantize _f16

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
    return (u & 0x80u) ? (int(u) - 256) : int(u);
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


    uint off = b * 2u;
    uint16_t h = uint16_t(g_Src[off]) | (uint16_t(g_Src[off + 1]) << 8);
    g_Dst[b] = f16_bits_to_f32(h);

}
