// Probe: read g_Params[3] (k) and g_Params[1] (m) and scatter them
// into the output: out[i] = (i < 16) ? m : k. Isolates the params
// buffer read in a cooperative-matrix pipeline.
StructuredBuffer<uint64_t> g_Params;
RWStructuredBuffer<half> g_Out;
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint idx = threadIdx.x;
    if (idx >= 256u) return;
    uint64_t m = g_Params[1];
    uint64_t k = g_Params[3];
    g_Out[idx] = half((idx < 16u) ? m : k);
}
