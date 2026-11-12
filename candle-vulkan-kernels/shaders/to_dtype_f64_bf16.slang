// to_dtype: F64 (double) -> BF16 (emulated u16).
import bf16_type;

StructuredBuffer<double> g_In;
RWStructuredBuffer<BF16> g_Out;
layout(push_constant) uint64_t g_Count;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Count)) { g_Out[idx] = bf16_encode(float(g_In[idx])); }
}
