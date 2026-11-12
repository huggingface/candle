// to_dtype: F16 -> I32. g_Out[i] = <expr> for i < g_Count.

import sat_cast;
StructuredBuffer<half> g_In;
RWStructuredBuffer<int32_t> g_Out;
layout(push_constant) uint64_t g_Count;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Count)) { g_Out[idx] = sat_f32_i32(float(g_In[idx])); }
}
