import f8e4m3_type;

StructuredBuffer<uint32_t> g_In;
RWStructuredBuffer<FP8> g_Out;
layout(push_constant) uint64_t g_Count;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Count)) { g_Out[idx] = f8e4m3_encode(float(float(g_In[idx]))); }
}
