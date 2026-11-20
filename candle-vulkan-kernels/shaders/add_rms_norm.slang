// add_rms_norm.slang
// Fused add + RMSNorm over the last dim, one 256-thread workgroup per row.
//
// residual[i][j] = x[i][j] + h[i][j]
// norm[i][j]     = residual[i][j] / m[i] * alpha[j]
// m[i]           = sqrt( sum_j(residual[i][j]^2) / n + eps )
//
// This fuses the residual add (x + h) and the following RMSNorm into a single
// dispatch, saving one kernel launch and one temporary buffer per decode step.
// Each thread folds a strided slice, thread 0 combines the partials in shared
// memory, and every thread writes its own residual and normalized slices.

import common_norm;

StructuredBuffer<float> g_X;
StructuredBuffer<float> g_H;
StructuredBuffer<float> g_Alpha;
RWStructuredBuffer<float> g_Residual;
RWStructuredBuffer<float> g_Norm;
layout(push_constant) Params g_Params;

static groupshared double gRed[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    uint i = groupID.x;
    uint tid = groupThreadID.x;
    if (i >= uint(g_Params.n_rows))
    {
        return;
    }
    uint64_t n = g_Params.n;
    uint64_t base = uint64_t(i) * n;
    double ps = 0.0;
    for (uint64_t j = tid; j < n; j += 256u)
    {
        float a = g_X[base + j] + g_H[base + j];
        g_Residual[base + j] = a;
        ps += double(a) * double(a);
    }
    gRed[tid] = ps;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        for (uint t = 1u; t < 256u; ++t)
        {
            gRed[0] = gRed[0] + gRed[t];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    double inv_m = 1.0 / (sqrt(gRed[0] / double(n) + g_Params.eps));
    float inv_m_f = float(inv_m);
    for (uint64_t j = tid; j < n; j += 256u)
    {
        g_Norm[base + j] = g_Residual[base + j] * inv_m_f * g_Alpha[j];
    }
}