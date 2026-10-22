// add_rms_norm.slang
// Fused add + RMSNorm over the last dim. One thread per row.
//
// residual[i][j] = x[i][j] + h[i][j]
// norm[i][j]     = residual[i][j] / m[i] * alpha[j]
// m[i]           = sqrt( sum_j(residual[i][j]^2) / n + eps )
//
// This fuses the residual add (x + h) and the following RMSNorm into a single
// dispatch, saving one kernel launch and one temporary buffer per decode step.

import common_norm;

StructuredBuffer<float> g_X;
StructuredBuffer<float> g_H;
StructuredBuffer<float> g_Alpha;
RWStructuredBuffer<float> g_Residual;
RWStructuredBuffer<float> g_Norm;
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
        float a = g_X[base + j] + g_H[base + j];
        g_Residual[base + j] = a;
        sum2 += double(a) * double(a);
    }
    double inv_m = 1.0 / (sqrt(sum2 / double(n) + g_Params.eps));
    float inv_m_f = float(inv_m);
    for (uint64_t j = 0; j < n; j += 1)
    {
        g_Norm[base + j] = g_Residual[base + j] * inv_m_f * g_Alpha[j];
    }
}
