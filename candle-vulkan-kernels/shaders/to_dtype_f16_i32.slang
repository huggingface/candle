// to_dtype: F16 -> I32. g_Out[i] = <expr> for i < g_Params[0].

StructuredBuffer<half> g_In;
RWStructuredBuffer<int32_t> g_Out;
StructuredBuffer<float> g_Params; // [0] = element count
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = int32_t(float(g_In[idx])); }
}
