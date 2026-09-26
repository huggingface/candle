// dst[i] = src[i] * mul + add
//
// Params: [0]=total, [1]=mul, [2]=add
StructuredBuffer<double> g_In;
RWStructuredBuffer<double> g_Out;
StructuredBuffer<float> g_Params;
[shader("compute")]
[numthreads(256, 1, 1)]
void main_affine(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i < uint(g_Params[0]))
    {
        g_Out[i] = g_In[i] * g_Params[1] + g_Params[2];
    }
}
