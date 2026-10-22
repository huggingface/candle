// to_dtype: BF16 (emulated u16) -> F32 (float).
import bf16_emulation;

StructuredBuffer<uint16_t> g_In;
RWStructuredBuffer<float> g_Out;
layout(push_constant) uint64_t g_Count;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Count)) { g_Out[idx] = bf16_decode(g_In[idx]); }
}
