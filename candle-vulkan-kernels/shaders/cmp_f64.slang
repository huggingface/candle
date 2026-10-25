// Comparison ops (Eq/Ne/Lt/Le/Gt/Ge): broadcast f64 inputs -> uint8 output
// (1 where the predicate holds, 0 otherwise), matching the CPU backend.
//
// Params layout: [0]=total, [1]=lhs_ndim, [2]=rhs_ndim,
//                [3..7]=lhs dims, [7..11]=rhs dims.
import common_cmp;
StructuredBuffer<double> g_In;
StructuredBuffer<double> g_Rhs;
RWStructuredBuffer<uint8_t> g_Out;
layout(push_constant) Params g_Params;

uint64_t rhs_offset(uint i)
{
    uint64_t rhs_ndim = g_Params.rhs_ndim;
    if (rhs_ndim == 0u) { return 0u; }
    uint64_t lhs_ndim = g_Params.lhs_ndim;
    uint64_t idx[4];
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
void main_eq(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = (g_In[idx] == g_Rhs[rhs_offset(idx)]) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_ne(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = (g_In[idx] != g_Rhs[rhs_offset(idx)]) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_lt(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = (g_In[idx] < g_Rhs[rhs_offset(idx)]) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_le(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = (g_In[idx] <= g_Rhs[rhs_offset(idx)]) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_gt(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = (g_In[idx] > g_Rhs[rhs_offset(idx)]) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_ge(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < g_Params.count)
    {
        g_Out[idx] = (g_In[idx] >= g_Rhs[rhs_offset(idx)]) ? (uint8_t)1u : (uint8_t)0u;
    }
}
