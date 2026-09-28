// RMSNorm over the last dim.
// out[i][j] = x[i][j] / m[i] * alpha[j]
// m[i] = sqrt(sum_j(x[i][j]^2) / n + eps)
StructuredBuffer<float> g_Src;
StructuredBuffer<float> g_Alpha;
RWStructuredBuffer<float> g_Dst;
StructuredBuffer<float> g_Params; // [0]=n_rows, [1]=n, [2]=eps
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
    float eps = g_Params[2];
    uint base = i * n;
    float sum2 = 0.0f;
    for (uint j = 0u; j < n; j += 1u)
    {
        float v = g_Src[base + j];
        sum2 += v * v;
    }
    float inv_m = 1.0f / sqrt(sum2 / float(n) + eps);
    for (uint j = 0u; j < n; j += 1u)
    {
        g_Dst[base + j] = g_Src[base + j] * inv_m * g_Alpha[j];
    }
}
