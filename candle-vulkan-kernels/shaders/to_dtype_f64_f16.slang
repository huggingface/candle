// to_dtype: F64 (double) -> F16 (half). g_Out[i] = half(g_In[i]) for i < g_Count.
StructuredBuffer<double> g_In;
RWStructuredBuffer<half> g_Out;
layout(push_constant) uint64_t g_Count;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Count)) { g_Out[idx] = half(g_In[idx]); }
}
