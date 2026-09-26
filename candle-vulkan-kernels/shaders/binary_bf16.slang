// Broadcast bf16 binary ops (add/sub/mul/div/maximum/minimum).
//
// bf16 is stored as its 16-bit bit pattern (uint16_t). Add/Sub/Mul/Div decode
// both operands to float, compute in float, and encode the result back to bf16,
// matching the half crate's `bf16` arithmetic: from_f32(to_f32(a) OP to_f32(b)).
// Maximum/Minimum compare the decoded floats and write back the *original* bit
// pattern of the winner (|v1 > v2 { v2 } else { v1 }), matching candle's CPU.
//
// Params layout: [0]=total, [1]=lhs_ndim, [2]=rhs_ndim,
//                [3..7]=lhs dims, [7..11]=rhs dims.
StructuredBuffer<uint16_t> g_In;
StructuredBuffer<uint16_t> g_Rhs;
RWStructuredBuffer<uint16_t> g_Out;
StructuredBuffer<float> g_Params;

// bf16 -> float (half crate bf16_to_f32).
float bf16_decode(uint16_t b)
{
    uint x = (uint)b;
    if ((x & 0x7FFFu) > 0x7F80u) { return bit_cast<float>(x | 0x40u); }
    return bit_cast<float>(x << 16);
}

// float -> bf16 (half crate f32_to_bf16, round-to-nearest-even).
uint16_t bf16_encode(float v)
{
    uint x = bit_cast<uint>(v);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) { return (uint16_t)((x >> 16) | 0x0040u); }
    uint rb = 0x8000u;
    if ((x & rb) != 0u && (x & (3u * rb - 1u)) != 0u) { return (uint16_t)((x >> 16) + 1u); }
    return (uint16_t)(x >> 16);
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
        float a = bf16_decode(g_In[idx]);
        float b = bf16_decode(g_Rhs[rhs_offset(idx)]);
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
        float a = bf16_decode(g_In[idx]);
        float b = bf16_decode(g_Rhs[rhs_offset(idx)]);
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
        g_Out[idx] = bf16_encode(bf16_decode(g_In[idx]) + bf16_decode(g_Rhs[rhs_offset(idx)]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sub(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = bf16_encode(bf16_decode(g_In[idx]) - bf16_decode(g_Rhs[rhs_offset(idx)]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_mul(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = bf16_encode(bf16_decode(g_In[idx]) * bf16_decode(g_Rhs[rhs_offset(idx)]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_div(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = bf16_encode(bf16_decode(g_In[idx]) / bf16_decode(g_Rhs[rhs_offset(idx)]));
    }
}
