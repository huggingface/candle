// Fills an i64 storage buffer with a constant value.
//
// Push constants: g_Params.count = element count, g_Params.value = fill value
// (the integer value stored as a float; truncated to i64).
import common_const_set;

RWStructuredBuffer<int64_t> g_Out;
layout(push_constant) Params g_Params;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params.count))
    {
        g_Out[idx] = int64_t(g_Params.value);
    }
}
