// to_dtype: F64 -> I64. g_Out[i] = <expr> for i < g_Count.

import sat_cast;
StructuredBuffer<double> g_In;
RWStructuredBuffer<int64_t> g_Out;
layout(push_constant) uint64_t g_Count;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Count)) { g_Out[idx] = sat_f64_i64(g_In[idx]); }
}
