// to_dtype: F32 -> F8E4M3. g_Out[i] = <expr> for i < g_Params[0].
// Shared dtype cast helpers (bf16 + f8e4m3 emulation), matching the microfloat/half crates.
float bf16_decode(uint16_t b)
{
    uint x = (uint)b;
    if ((x & 0x7FFFu) > 0x7F80u) { return bit_cast<float>(x | 0x40u); }
    return bit_cast<float>(x << 16);
}
uint16_t bf16_encode(float v)
{
    uint x = bit_cast<uint>(v);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) { return (uint16_t)((x >> 16) | 0x0040u); }
    uint rb = 0x8000u;
    if ((x & rb) != 0u && (x & (3u * rb - 1u)) != 0u) { return (uint16_t)((x >> 16) + 1u); }
    return (uint16_t)(x >> 16);
}
float f8e4m3_f2i(int e)
{
    if (e < -149) { return 0.0f; }
    if (e > 127) { return bit_cast<float>(0x7f800000u); }
    if (e >= -126) { return bit_cast<float>(uint(e + 127) << 23); }
    return bit_cast<float>(1u << uint(e + 149));
}
uint f8e4m3_rrs(uint64_t value, int shift)
{
    if (shift >= 64) { return 0u; }
    uint s = uint(shift);
    uint q = (uint)(value >> s);
    uint rem = (uint)(value & (((uint64_t)1u << s) - (uint64_t)1u));
    uint half = 1u << (s - 1u);
    if (rem > half || (rem == half && (q & 1u) == 1u)) { return q + 1u; }
    return q;
}
uint64_t f8e4m3_rti(float value, int scale_exp)
{
    uint bits = bit_cast<uint>(value);
    int exp = int((bits >> 23) & 0xffu);
    uint fraction = bits & 0x7f_ffffu;
    uint64_t sig;
    int power;
    if (exp == 0) { sig = (uint64_t)fraction; power = -149; }
    else { sig = (uint64_t)((1u << 23) | fraction); power = exp - 150; }
    int shift = power - scale_exp;
    if (shift >= 0) { return sig << uint(shift); }
    return (uint64_t)f8e4m3_rrs(sig, -shift);
}
int f8e4m3_fl2(float v)
{
    uint bits = bit_cast<uint>(v) & 0x7fff_ffffu;
    int exp = int((bits >> 23) & 0xffu);
    if (exp == 0) { return -200; }
    return exp - 127;
}
float f8e4m3_decode(uint8_t bits)
{
    uint b = (uint)bits;
    bool neg = (b & 0x80u) != 0u;
    if ((b & 0x78u) == 0x78u)
    {
        if ((b & 0x07u) != 0u)
        {
            return bit_cast<float>(neg ? 0xffc00000u : 0x7fc00000u);
        }
        return bit_cast<float>(neg ? 0xff800000u : 0x7f800000u);
    }
    uint mag = b & 0x7fu;
    float sign = neg ? -1.0f : 1.0f;
    if (mag == 0u) { return (sign < 0.0f) ? -0.0f : 0.0f; }
    uint exp = (b & 0x78u) >> 3;
    float mant = (float)(b & 0x07u);
    float scale = (exp == 0u) ? f8e4m3_f2i(1 - 7) : f8e4m3_f2i(int(exp) - 7);
    float significand = (exp == 0u) ? (mant / f8e4m3_f2i(3)) : (1.0f + mant / f8e4m3_f2i(3));
    return sign * significand * scale;
}
uint8_t f8e4m3_encode(float v)
{
    uint x = bit_cast<uint>(v);
    uint mag = x & 0x7fff_ffffu;
    bool neg = (x & 0x80000000u) != 0u;
    if (mag > 0x7f800000u)
    {
        uint base = 0x78u | (1u << 2);
        return (uint8_t)(neg ? (base | 0x80u) : base);
    }
    if (mag == 0x7f800000u) { return (uint8_t)(neg ? 0xf8u : 0x78u); }
    float abs = bit_cast<float>(mag);
    if (abs == 0.0f) { return (uint8_t)(neg ? 0x80u : 0x00u); }
    if (abs >= 248.0f) { return (uint8_t)(neg ? 0xf8u : 0x78u); }
    int exp = f8e4m3_fl2(abs);
    uint bits;
    if (exp < -6)
    {
        uint64_t scaled = f8e4m3_rti(abs, -9);
        bits = (uint)(scaled < (uint64_t)8u ? scaled : (uint64_t)8u);
    }
    else
    {
        int ve = exp;
        uint64_t sig = f8e4m3_rti(abs, ve - 3);
        if (sig >= (uint64_t)16u) { sig >>= 1; ve += 1; }
        uint exponent = uint(ve + 7);
        uint mantissa = (uint)(sig - (uint64_t)8u);
        bits = (exponent << 3) | mantissa;
    }
    return (uint8_t)(neg ? (bits | 0x80u) : bits);
}


StructuredBuffer<float> g_In;
RWStructuredBuffer<uint8_t> g_Out;
StructuredBuffer<float> g_Params; // [0] = element count
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = f8e4m3_encode(float(g_In[idx])); }
}
