import bf16_emulation;
import f8e4m3_emulation;

StructuredBuffer<uint8_t> g_In;
RWStructuredBuffer<int16_t> g_Out;
layout(push_constant) uint64_t g_Count;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Count)) { g_Out[idx] = int16_t(f8e4m3_decode(g_In[idx])); }
}
