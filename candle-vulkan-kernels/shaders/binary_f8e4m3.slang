// Broadcast f8e4m3 binary ops (add/sub/mul/div/maximum/minimum).
//
// f8e4m3 (candle's F8E4M3) is the microfloat `f8e4m3` format: 1 sign, 4
// exponent (bias 7), 3 mantissa bits, IEEE-style NaN and infinity, max finite
// 240, overflow -> infinity. It is stored as its 8-bit bit pattern (uint8_t).
// Add/Sub/Mul/Div decode both operands to float, compute in float, and encode
// the result back, matching the microfloat crate:
//   from_f32(to_f32(a) OP to_f32(b))
// Maximum/Minimum compare the decoded floats and write back the *original* bit
// pattern of the winner (|v1 < v2 { v2 } else { v1 }), matching candle's CPU.
//
// Params layout: [0]=total, [1]=lhs_ndim, [2]=rhs_ndim,
//                [3..7]=lhs dims, [7..11]=rhs dims.
StructuredBuffer<uint8_t> g_In;
StructuredBuffer<uint8_t> g_Rhs;
RWStructuredBuffer<uint8_t> g_Out;
StructuredBuffer<float> g_Params;

// 2^e as float (microfloat exp2i), e in the f32-representable range.
float f8e4m3_f2i(int e)
{
    if (e < -149) { return 0.0f; }
    if (e > 127) { return bit_cast<float>(0x7f800000u); }
    if (e >= -126) { return bit_cast<float>(uint(e + 127) << 23); }
    return bit_cast<float>(1u << uint(e + 149));
}

// Round-to-nearest-even right shift (microfloat round_right_shift).
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

// Round value * 2^scale_exp to the nearest integer (microfloat round_to_integer).
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

// floor(log2(v)) for positive finite v (microfloat floor_log2_f32).
int f8e4m3_fl2(float v)
{
    uint bits = bit_cast<uint>(v) & 0x7fff_ffffu;
    int exp = int((bits >> 23) & 0xffu);
    if (exp == 0) { return -200; }
    return exp - 127;
}

// f8e4m3 bit pattern -> float (microfloat decode_f32, bias 7, e4m3).
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

// float -> f8e4m3 bit pattern (microfloat encode_f32, bias 7, e4m3).
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

uint rhs_offset(uint i)
{
    uint rhs_ndim = uint(g_Params[2]);
    if (rhs_ndim == 0u) { return 0u; }
    uint lhs_ndim = uint(g_Params[1]);
    uint idx[4];
    uint rem = i;
    for (uint d = 0u; d < lhs_ndim; d += 1u)
    {
        uint dim = uint(g_Params[3u + (lhs_ndim - 1u - d)]);
        idx[lhs_ndim - 1u - d] = rem % dim;
        rem = rem / dim;
    }
    uint off = 0u;
    for (uint j = 0u; j < rhs_ndim; j += 1u)
    {
        uint r = uint(g_Params[7u + j]);
        uint lidx = idx[lhs_ndim - rhs_ndim + j];
        off = off * r + (r != 1u ? lidx : 0u);
    }
    return off;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_maximum(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        float a = f8e4m3_decode(g_In[idx]);
        float b = f8e4m3_decode(g_Rhs[rhs_offset(idx)]);
        g_Out[idx] = (a < b) ? g_Rhs[rhs_offset(idx)] : g_In[idx];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_minimum(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        float a = f8e4m3_decode(g_In[idx]);
        float b = f8e4m3_decode(g_Rhs[rhs_offset(idx)]);
        g_Out[idx] = (a > b) ? g_Rhs[rhs_offset(idx)] : g_In[idx];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_add(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = f8e4m3_encode(f8e4m3_decode(g_In[idx]) + f8e4m3_decode(g_Rhs[rhs_offset(idx)]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sub(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = f8e4m3_encode(f8e4m3_decode(g_In[idx]) - f8e4m3_decode(g_Rhs[rhs_offset(idx)]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_mul(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = f8e4m3_encode(f8e4m3_decode(g_In[idx]) * f8e4m3_decode(g_Rhs[rhs_offset(idx)]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_div(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = f8e4m3_encode(f8e4m3_decode(g_In[idx]) / f8e4m3_decode(g_Rhs[rhs_offset(idx)]));
    }
}
