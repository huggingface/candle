import bf16_type;
import sat_cast;

StructuredBuffer<BF16> g_In;
RWStructuredBuffer<int64_t> g_Out;
layout(push_constant) uint64_t g_Count;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Count)) { g_Out[idx] = sat_f32_i64(bf16_decode(g_In[idx])); }
}
