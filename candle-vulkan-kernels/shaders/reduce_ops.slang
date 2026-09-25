// Last-axis reductions not covered by the WGSL kernels:
// out[row] = min_c in[row*cols + c]                      (main_reduce_min)
// out[row] = argmin_c in[row*cols + c]                   (main_reduce_argmin)
// out[row] = argmax_c in[row*cols + c]                   (main_reduce_argmax)
StructuredBuffer<float> g_In;
RWStructuredBuffer<float> g_OutF;
RWStructuredBuffer<uint> g_OutI;
StructuredBuffer<float> g_Params; // [0]=rows, [1]=cols

[shader("compute")]
[numthreads(1, 1, 1)]
void main_reduce_min(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint row = dispatchThreadID.x;
    uint rows = uint(g_Params[0]);
    uint cols = uint(g_Params[1]);
    if (row >= rows || cols == 0u)
    {
        return;
    }
    float acc = g_In[row * cols];
    for (uint c = 1u; c < cols; c += 1u)
    {
        float v = g_In[row * cols + c];
        if (v < acc)
        {
            acc = v;
        }
    }
    g_OutF[row] = acc;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main_reduce_argmin(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint row = dispatchThreadID.x;
    uint rows = uint(g_Params[0]);
    uint cols = uint(g_Params[1]);
    if (row >= rows || cols == 0u)
    {
        return;
    }
    float best = g_In[row * cols];
    uint besti = 0u;
    for (uint c = 1u; c < cols; c += 1u)
    {
        float v = g_In[row * cols + c];
        if (v < best)
        {
            best = v;
            besti = c;
        }
    }
    g_OutI[row] = besti;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main_reduce_argmax(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint row = dispatchThreadID.x;
    uint rows = uint(g_Params[0]);
    uint cols = uint(g_Params[1]);
    if (row >= rows || cols == 0u)
    {
        return;
    }
    float best = g_In[row * cols];
    uint besti = 0u;
    for (uint c = 1u; c < cols; c += 1u)
    {
        float v = g_In[row * cols + c];
        if (v > best)
        {
            best = v;
            besti = c;
        }
    }
    g_OutI[row] = besti;
}
