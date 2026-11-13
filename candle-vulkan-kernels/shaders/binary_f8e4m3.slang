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
//                [3..11]=lhs dims, [11..19]=rhs dims,
//                [19]=lhs_data_ndim, [20..28]=lhs_data_dims.
import f8e4m3_type;
import common_binary;

StructuredBuffer<FP8> g_In;
StructuredBuffer<FP8> g_Rhs;
RWStructuredBuffer<FP8> g_Out;
layout(push_constant) Params g_Params;

uint64_t rhs_offset(uint i)
{
    uint64_t rhs_ndim = g_Params.rhs_ndim;
    if (rhs_ndim == 0u) { return 0u; }
    uint64_t lhs_ndim = g_Params.lhs_ndim;
    uint64_t idx[8];
    uint64_t rem = i;
    for (uint d = 0u; d < lhs_ndim; d += 1u)
    {
        uint64_t dim = g_Params.lhs_dims[(lhs_ndim - 1u - d)];
        idx[lhs_ndim - 1u - d] = rem % dim;
        rem = rem / dim;
    }
    uint64_t off = 0u;
    for (uint j = 0u; j < rhs_ndim; j += 1u)
    {
        uint64_t r = g_Params.rhs_dims[j];
        uint64_t lidx = idx[lhs_ndim - rhs_ndim + j];
        off = off * r + (r != 1u ? lidx : 0u);
    }
    return off;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_maximum(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
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
    if (idx < g_Params.count)
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
    if (idx < g_Params.count)
    {
        g_Out[idx] = f8e4m3_encode(f8e4m3_decode(g_In[idx]) + f8e4m3_decode(g_Rhs[rhs_offset(idx)]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_sub(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = f8e4m3_encode(f8e4m3_decode(g_In[idx]) - f8e4m3_decode(g_Rhs[rhs_offset(idx)]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_mul(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = f8e4m3_encode(f8e4m3_decode(g_In[idx]) * f8e4m3_decode(g_Rhs[rhs_offset(idx)]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_div(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = f8e4m3_encode(f8e4m3_decode(g_In[idx]) / f8e4m3_decode(g_Rhs[rhs_offset(idx)]));
    }
}
