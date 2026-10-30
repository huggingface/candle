// Fills a f64 (double) storage buffer with a constant value.
//
// Push constants: g_Params.count = element count, g_Params.value = fill value
// (an f32; widened to f64).
import common_const_set;

RWStructuredBuffer<double> g_Out;
layout(push_constant) Params g_Params;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        g_Out[idx] = double(g_Params.value);
    }
}
