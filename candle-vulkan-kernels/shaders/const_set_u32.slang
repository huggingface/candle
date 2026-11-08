// Fills a u32 storage buffer with a constant value.
//
// Push constants: g_Params.count = element count, g_Params.value = the fill
// value (uint32_t, carried natively to avoid an f32 round-trip).
import common_const_set;

RWStructuredBuffer<uint> g_Out;
layout(push_constant) U32Params g_Params;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        g_Out[idx] = g_Params.value;
    }
}
