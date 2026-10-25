// Softmax over the last dim.
// out[i][j] = exp(x[i][j] - max) / sum_k(exp(x[i][k] - max))
import common_softmax;
StructuredBuffer<float> g_Src;
RWStructuredBuffer<float> g_Dst;
layout(push_constant) Params g_Params;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    uint64_t n_rows = g_Params.n_rows;
    if (i >= n_rows)
    {
        return;
    }
    uint64_t n = g_Params.n;
    uint64_t base = i * n;
    float mx = g_Src[base];
    for (uint j = 1u; j < n; j += 1u)
    {
        if (g_Src[base + j] > mx)
        {
            mx = g_Src[base + j];
        }
    }
    float sum = 0.0f;
    for (uint j = 0u; j < n; j += 1u)
    {
        float e = exp(g_Src[base + j] - mx);
        g_Dst[base + j] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (uint j = 0u; j < n; j += 1u)
    {
        g_Dst[base + j] = g_Dst[base + j] * inv;
    }
}
