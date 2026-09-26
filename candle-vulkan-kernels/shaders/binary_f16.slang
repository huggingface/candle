// Broadcast f16 binary ops (add/sub/mul/div/maximum/minimum).
//
// Params layout: [0]=total, [1]=lhs_ndim, [2]=rhs_ndim,
//                [3..7]=lhs dims, [7..11]=rhs dims.
// The rhs is right-aligned against the lhs; a rhs dim of 1 is broadcast.
StructuredBuffer<half> g_In;
StructuredBuffer<half> g_Rhs;
RWStructuredBuffer<half> g_Out;
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
        // Decompose right to left over the lhs dimensions.
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
        g_Out[idx] = max(g_In[idx], g_Rhs[rhs_offset(idx)]);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_minimum(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = min(g_In[idx], g_Rhs[rhs_offset(idx)]);
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_add(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = g_In[idx] + g_Rhs[rhs_offset(idx)];
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_sub(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = g_In[idx] - g_Rhs[rhs_offset(idx)];
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_mul(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = g_In[idx] * g_Rhs[rhs_offset(idx)];
    }
}
[shader("compute")]
[numthreads(256, 1, 1)]
void main_div(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0]))
    {
        g_Out[idx] = g_In[idx] / g_Rhs[rhs_offset(idx)];
    }
}
