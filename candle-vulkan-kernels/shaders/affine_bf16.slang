// affine (dst = src*mul + add) for bf16 (emulated): decode to float, compute, encode back.
import bf16_emulation;
import common_affine;

StructuredBuffer<uint16_t> g_In;
RWStructuredBuffer<uint16_t> g_Out;
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main_affine(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i < uint(g_Params.total))
    {
        float x = bf16_decode(g_In[i]);
        g_Out[i] = bf16_encode(float(double(x) * g_Params.mul + g_Params.add));
    }
}
