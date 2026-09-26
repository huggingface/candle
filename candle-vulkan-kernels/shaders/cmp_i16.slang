// Comparison ops (Eq/Ne/Lt/Le/Gt/Ge): broadcast i16 inputs -> uint8 output
// (1 where the predicate holds, 0 otherwise), matching the CPU backend.
//
// Params layout: [0]=total, [1]=lhs_ndim, [2]=rhs_ndim,
//                [3..7]=lhs dims, [7..11]=rhs dims.
StructuredBuffer<int16_t> g_In;
StructuredBuffer<int16_t> g_Rhs;
RWStructuredBuffer<uint8_t> g_Out;
StructuredBuffer<float> g_Params;

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
        g_Out[idx] = (g_In[idx] == g_Rhs[rhs_offset(idx)]) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_ne(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = (g_In[idx] != g_Rhs[rhs_offset(idx)]) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_lt(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = (g_In[idx] < g_Rhs[rhs_offset(idx)]) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_le(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = (g_In[idx] <= g_Rhs[rhs_offset(idx)]) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_gt(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = (g_In[idx] > g_Rhs[rhs_offset(idx)]) ? (uint8_t)1u : (uint8_t)0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_ge(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = (g_In[idx] >= g_Rhs[rhs_offset(idx)]) ? (uint8_t)1u : (uint8_t)0u;
    }
}
