// LayerNorm over the last dim.
// out[i][j] = (x[i][j] - mean[i]) * inv_std[i] * alpha[j] + beta[j]
// mean[i] = sum_j(x[i][j]) / n
// var[i] = sum_j(x[i][j]^2) / n - mean[i]^2
// inv_std[i] = 1 / sqrt(var[i] + eps)
StructuredBuffer<float> g_Src;
StructuredBuffer<float> g_Alpha;
StructuredBuffer<float> g_Beta;
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
    float sum = 0.0f;
    float sum2 = 0.0f;
    for (uint j = 0u; j < n; j += 1u)
    {
        float v = g_Src[base + j];
        sum += v;
        sum2 += v * v;
    }
    float mean = sum / float(n);
    float var = sum2 / float(n) - mean * mean;
    float inv_std = 1.0f / sqrt(var + eps);
    for (uint j = 0u; j < n; j += 1u)
    {
        float x = g_Src[base + j];
        g_Dst[base + j] = (x - mean) * inv_std * g_Alpha[j] + g_Beta[j];
    }
}
