// Fills a float storage buffer with a constant value.
//
// Push constants: g_Params.count = element count, g_Params.value = fill value.
RWStructuredBuffer<float> g_Out;
struct Params
{
    uint64_t count;
    float value;
};
layout(push_constant) Params g_Params;
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
