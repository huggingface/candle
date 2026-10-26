// post_rms_norm_add.slang
// Fused RMSNorm + residual add over the last dim. One thread per row.
//
// out[i][j] = x[i][j] + h[i][j] / m[i] * alpha[j]
// m[i]      = sqrt( sum_j(h[i][j]^2) / n + eps )
//
// Gemma-3 style block: the sublayer output `h` is normalized first and
// then added to the residual `x`. Fusing both into a single dispatch
// removes the intermediate norm output buffer and one kernel launch per
// decode step.

import common_norm;

StructuredBuffer<float> g_X;
StructuredBuffer<float> g_H;
StructuredBuffer<float> g_Alpha;
RWStructuredBuffer<float> g_Out;
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
    double sum2 = 0.0;
    for (uint64_t j = 0; j < n; j += 1)
    {
        float a = g_H[base + j];
        sum2 += double(a) * double(a);
    }
    double inv_m = 1.0 / (sqrt(sum2 / double(n) + g_Params.eps));
    float inv_m_f = float(inv_m);
    for (uint64_t j = 0; j < n; j += 1)
    {
        g_Out[base + j] = g_X[base + j] + g_H[base + j] * inv_m_f * g_Alpha[j];
    }
}
