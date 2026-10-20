// Sigmoid: out[i] = 1 / (1 + exp(-x[i]))
StructuredBuffer<float> g_Src;
RWStructuredBuffer<float> g_Dst;
StructuredBuffer<uint64_t> g_Params; // [0] = element count
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    uint64_t count = g_Params[0];
    if (i >= count)
    {
        return;
    }
    float x = g_Src[i];
    g_Dst[i] = 1.0f / (1.0f + exp(-x));
}
