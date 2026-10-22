// to_dtype: F32 (float) -> BF16 (emulated u16).
import bf16_emulation;

StructuredBuffer<float> g_In;
RWStructuredBuffer<uint16_t> g_Out;
layout(push_constant) uint64_t g_Count;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Count)) { g_Out[idx] = bf16_encode(g_In[idx]); }
}
