// Fills a bf16 (emulated u16) storage buffer with a constant value.
//
// Push constants: g_Params.count = element count, g_Params.value = fill value.
import bf16_emulation;
import common_const_set;

RWStructuredBuffer<uint16_t> g_Out;
layout(push_constant) Params g_Params;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        g_Out[idx] = bf16_encode(g_Params.value);
    }
}
