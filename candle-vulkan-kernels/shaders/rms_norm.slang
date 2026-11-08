// RMSNorm over the last dim, one 256-thread workgroup per row.
// out[i][j] = x[i][j] / m[i] * alpha[j]
// m[i] = sqrt(sum_j(x[i][j]^2) / n + eps)
//
// Each thread folds a strided slice of the row's squared sum, thread 0
// combines the partials in shared memory (the uniform-barrier pattern, since a
// divergent tree reduction miscompiles on RADV), and every thread normalizes
// its own slice. This shader is compiled with `-fp-mode precise` (see
// `PRECISE_FP_SHADERS` in build.rs) so each `sq = v * v` is a separately
// rounded operation from the `+=`.
import common_norm;

StructuredBuffer<float> g_Src;
StructuredBuffer<float> g_Alpha;
RWStructuredBuffer<float> g_Dst;
layout(push_constant) Params g_Params;

static groupshared float gRed[256];

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
    float ps = 0.0f;
    for (uint64_t j = tid; j < n; j += 256u)
    {
        float v = g_Src[base + j];
        float sq = v * v;
        ps += sq;
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
    // Match the reference normalization: the sum is scaled by 1/n in f64, the
    // epsilon add is a separate f64 step, then f32 sqrt.
    float normed = float(double(gRed[0]) * (1.0 / double(n)));
    float m = sqrt(float(double(normed) + g_Params.eps));
    for (uint64_t j = tid; j < n; j += 256u)
    {
        g_Dst[base + j] = (g_Src[base + j] / m) * g_Alpha[j];
    }
}