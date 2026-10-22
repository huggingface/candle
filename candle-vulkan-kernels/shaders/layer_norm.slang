// LayerNorm over the last dim.
// out[i][j] = (x[i][j] - mean[i]) * inv_std[i] * alpha[j] + beta[j]
// mean[i] = sum_j(x[i][j]) / n
// var[i] = sum_j(x[i][j]^2) / n - mean[i]^2
// inv_std[i] = 1 / sqrt(var[i] + eps)
import common_norm;

StructuredBuffer<float> g_Src;
StructuredBuffer<float> g_Alpha;
StructuredBuffer<float> g_Beta;
RWStructuredBuffer<float> g_Dst;
layout(push_constant) Params g_Params;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i >= uint(g_Params.n_rows))
    {
        return;
    }
    uint64_t n = g_Params.n;
    uint64_t base = uint64_t(i) * n;
    float sum = 0.0f;
    float sum2 = 0.0f;
    for (uint64_t j = 0; j < n; j += 1)
    {
        float v = g_Src[base + j];
        sum += v;
        sum2 += v * v;
    }
    double mean = double(sum) / double(n);
    double var = double(sum2) / double(n) - mean * mean;
    double inv_std = 1.0 / sqrt(var + g_Params.eps);
    float mean_f = float(mean);
    float inv_std_f = float(inv_std);
    for (uint64_t j = 0; j < n; j += 1)
    {
        float x = g_Src[base + j];
        g_Dst[base + j] = (x - mean_f) * inv_std_f * g_Alpha[j] + g_Beta[j];
    }
}
