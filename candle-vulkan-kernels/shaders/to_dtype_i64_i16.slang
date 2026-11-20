// to_dtype: I64 -> I16. g_Out[i] = <expr> for i < g_Count.

StructuredBuffer<int64_t> g_In;
RWStructuredBuffer<int16_t> g_Out;
layout(push_constant) uint64_t g_Count;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Count)) { g_Out[idx] = int16_t(g_In[idx]); }
}
