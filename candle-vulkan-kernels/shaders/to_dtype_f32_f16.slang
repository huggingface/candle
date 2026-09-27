// to_dtype: F32 (float) -> F16 (half). g_Out[i] = half(g_In[i]) for i < g_Params[0].
StructuredBuffer<float> g_In;
RWStructuredBuffer<half> g_Out;
StructuredBuffer<float> g_Params; // [0] = element count
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = half(g_In[idx]); }
}
