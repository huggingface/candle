// Fills an f8e4m3 (FP8) storage buffer with a constant value (emulated).
//
// Push constants: g_Params.count = element count, g_Params.value = fill value
// (an f32; encoded to f8e4m3).
import common_const_set;
import f8e4m3_type;

RWStructuredBuffer<FP8> g_Out;
layout(push_constant) Params g_Params;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        g_Out[idx] = f8e4m3_encode(g_Params.value);
    }
}
