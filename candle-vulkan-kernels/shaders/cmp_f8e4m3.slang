// Broadcast f8e4m3 comparison ops (eq/ne/lt/le/gt/ge): f8e4m3 inputs (stored as
// uint8_t bit patterns) -> uint8 output (1 where the predicate holds, 0
// otherwise), matching the CPU backend. Both operands are decoded to float and
// compared with IEEE semantics (NaN comparisons are false, except `ne`).
//
// f8e4m3 is the microfloat `f8e4m3` format: 1 sign, 4 exponent (bias 7),
// 3 mantissa bits, IEEE-style NaN and infinity, max finite 240.
//
// Params layout: [0]=total, [1]=lhs_ndim, [2]=rhs_ndim,
//                [3..7]=lhs dims, [7..11]=rhs dims.
StructuredBuffer<uint8_t> g_In;
StructuredBuffer<uint8_t> g_Rhs;
RWStructuredBuffer<uint8_t> g_Out;
StructuredBuffer<float> g_Params;

float f8e4m3_f2i(int e)
{
    if (e < -149) { return 0.0f; }
    if (e > 127) { return bit_cast<float>(0x7f800000u); }
    if (e >= -126) { return bit_cast<float>(uint(e + 127) << 23); }
    return bit_cast<float>(1u << uint(e + 149));
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
void main_eq(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        float a = f8e4m3_decode(g_In[idx]);
        float b = f8e4m3_decode(g_Rhs[rhs_offset(idx)]);
        g_Out[idx] = (a == b) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_ne(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        float a = f8e4m3_decode(g_In[idx]);
        float b = f8e4m3_decode(g_Rhs[rhs_offset(idx)]);
        g_Out[idx] = (a != b) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_lt(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        float a = f8e4m3_decode(g_In[idx]);
        float b = f8e4m3_decode(g_Rhs[rhs_offset(idx)]);
        g_Out[idx] = (a < b) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_le(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        float a = f8e4m3_decode(g_In[idx]);
        float b = f8e4m3_decode(g_Rhs[rhs_offset(idx)]);
        g_Out[idx] = (a <= b) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_gt(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        float a = f8e4m3_decode(g_In[idx]);
        float b = f8e4m3_decode(g_Rhs[rhs_offset(idx)]);
        g_Out[idx] = (a > b) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_ge(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        float a = f8e4m3_decode(g_In[idx]);
        float b = f8e4m3_decode(g_Rhs[rhs_offset(idx)]);
        g_Out[idx] = (a >= b) ? (uint8_t)1u : (uint8_t)0u;
    }
}
