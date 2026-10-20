// RMSNorm over the last dim.
// out[i][j] = x[i][j] / m[i] * alpha[j]
// m[i] = sqrt(sum_j(x[i][j]^2) / n + eps)
StructuredBuffer<float> g_Src;
StructuredBuffer<float> g_Alpha;
RWStructuredBuffer<float> g_Dst;
struct Params
{
    uint64_t n_rows;
    uint64_t n;
    double eps;
};
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
    float sum2 = 0.0f;
    for (uint64_t j = 0; j < n; j += 1)
    {
        float v = g_Src[base + j];
        sum2 += v * v;
    }
    double inv_m = 1.0 / sqrt(double(sum2) / double(n) + g_Params.eps);
    float inv_m_f = float(inv_m);
    for (uint64_t j = 0; j < n; j += 1)
    {
        g_Dst[base + j] = g_Src[base + j] * inv_m_f * g_Alpha[j];
    }
}
