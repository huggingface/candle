// Softmax over the last dim.
// out[i][j] = exp(x[i][j] - max) / sum_k(exp(x[i][k] - max))
StructuredBuffer<float> g_Src;
RWStructuredBuffer<float> g_Dst;
StructuredBuffer<float> g_Params; // [0]=n_rows, [1]=n
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    uint n_rows = uint(g_Params[0]);
    if (i >= n_rows)
    {
        return;
    }
    uint n = uint(g_Params[1]);
    uint base = i * n;
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
