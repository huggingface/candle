// to_dtype: F64 (double) -> F32 (float). g_Out[i] = float(g_In[i]) for i < g_Params[0].
StructuredBuffer<double> g_In;
RWStructuredBuffer<float> g_Out;
StructuredBuffer<float> g_Params; // [0] = element count
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[0])) { g_Out[idx] = float(g_In[idx]); }
}
