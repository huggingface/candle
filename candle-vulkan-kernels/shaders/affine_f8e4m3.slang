// affine (dst = src*mul + add) for f8e4m3 (emulated): decode to float, compute, encode back.
import common_affine;
import f8e4m3_type;

StructuredBuffer<FP8> g_In;
RWStructuredBuffer<FP8> g_Out;
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main_affine(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i < uint(g_Params.total))
    {
        float x = f8e4m3_decode(g_In[i]);
        g_Out[i] = f8e4m3_encode(float(double(x) * g_Params.mul + g_Params.add));
    }
}
