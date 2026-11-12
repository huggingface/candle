// dst[i] = src[i] * mul + add
//
// Params: [0]=total, [1]=mul, [2]=add
import common_affine;

StructuredBuffer<float> g_In;
RWStructuredBuffer<float> g_Out;
layout(push_constant) Params g_Params;
[shader("compute")]
[numthreads(256, 1, 1)]
void main_affine(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i < uint(g_Params.total))
    {
        g_Out[i] = float(double(g_In[i]) * g_Params.mul + g_Params.add);
    }
}
