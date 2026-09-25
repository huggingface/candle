// Fills a float storage buffer with a constant value.
//
// Params buffer layout: [0] = fill value, [1] = element count.
RWStructuredBuffer<float> g_Out;
StructuredBuffer<float> g_Params;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx < uint(g_Params[1]))
    {
        g_Out[idx] = g_Params[0];
    }
}
