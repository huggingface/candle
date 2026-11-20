// Broadcast u8 binary ops (add/sub/mul/div/maximum/minimum).
//
// Params layout: [0]=total, [1]=lhs_ndim, [2]=rhs_ndim,
//                [3..11]=lhs dims, [11..19]=rhs dims,
//                [19]=lhs_data_ndim, [20..28]=lhs_data_dims.
// The rhs is right-aligned against the lhs; a rhs dim of 1 is broadcast.
import common_binary;
StructuredBuffer<uint8_t> g_In;
StructuredBuffer<uint8_t> g_Rhs;
RWStructuredBuffer<uint8_t> g_Out;
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
        // Decompose right to left over the lhs dimensions.
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
        g_Out[idx] = max(g_In[idx], g_Rhs[rhs_offset(idx)]);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_minimum(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = min(g_In[idx], g_Rhs[rhs_offset(idx)]);
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_add(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = g_In[idx] + g_Rhs[rhs_offset(idx)];
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_sub(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = g_In[idx] - g_Rhs[rhs_offset(idx)];
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_mul(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = g_In[idx] * g_Rhs[rhs_offset(idx)];
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_div(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = g_In[idx] / g_Rhs[rhs_offset(idx)];
    }
}
