// Last-axis reductions over int64_t input:
// out[row] = sum_c in[row*cols + c]                      (main_reduce_sum)
// out[row] = max_c in[row*cols + c]                      (main_reduce_max)
// out[row] = min_c in[row*cols + c]                      (main_reduce_min)
// out[row] = argmin_c in[row*cols + c]                   (main_reduce_argmin)
// out[row] = argmax_c in[row*cols + c]                   (main_reduce_argmax)
//
// Sums and min/max/argmin/argmax comparisons run in the native int64_t type:
// an F64 round-trip would round I64 values above 2^53 (and collapse distinct
// values there), losing exactness for large integer reductions.
import common_reduce;
StructuredBuffer<int64_t> g_In;
RWStructuredBuffer<int64_t> g_OutF;
RWStructuredBuffer<uint> g_OutI;
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(1, 1, 1)]
void main_reduce_min(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint row = dispatchThreadID.x;
    uint64_t rows = g_Params.rows;
    uint64_t cols = g_Params.cols;
    if (row >= rows || cols == 0u)
    {
        return;
    }
    int64_t acc = g_In[row * cols];
    for (uint c = 1u; c < cols; c += 1u)
    {
        int64_t v = g_In[row * cols + c];
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
    uint64_t rows = g_Params.rows;
    uint64_t cols = g_Params.cols;
    if (row >= rows || cols == 0u)
    {
        return;
    }
    int64_t best = g_In[row * cols];
    uint besti = 0u;
    for (uint c = 1u; c < cols; c += 1u)
    {
        int64_t v = g_In[row * cols + c];
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
    uint64_t rows = g_Params.rows;
    uint64_t cols = g_Params.cols;
    if (row >= rows || cols == 0u)
    {
        return;
    }
    int64_t best = g_In[row * cols];
    uint besti = 0u;
    for (uint c = 1u; c < cols; c += 1u)
    {
        int64_t v = g_In[row * cols + c];
        if (v > best)
        {
            best = v;
            besti = c;
        }
    }
    g_OutI[row] = besti;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main_reduce_sum(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint row = dispatchThreadID.x;
    uint64_t rows = g_Params.rows;
    uint64_t cols = g_Params.cols;
    if (row >= rows || cols == 0u)
    {
        return;
    }
    int64_t acc = int64_t(0);
    for (uint c = 0u; c < cols; c += 1u)
    {
        acc += g_In[row * cols + c];
    }
    g_OutF[row] = acc;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main_reduce_max(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint row = dispatchThreadID.x;
    uint64_t rows = g_Params.rows;
    uint64_t cols = g_Params.cols;
    if (row >= rows || cols == 0u)
    {
        return;
    }
    int64_t acc = g_In[row * cols];
    for (uint c = 1u; c < cols; c += 1u)
    {
        int64_t v = g_In[row * cols + c];
        if (v > acc)
        {
            acc = v;
        }
    }
    g_OutF[row] = acc;
}