// LayerNorm over the last dim, one 256-thread workgroup per row.
// out[i][j] = (x[i][j] - mean[i]) * inv_std[i] * alpha[j] + beta[j]
// mean[i] = sum_j(x[i][j]) / n
// var[i] = sum_j(x[i][j]^2) / n - mean[i]^2
// inv_std[i] = 1 / sqrt(var[i] + eps)
//
// Each thread folds a strided slice into partial sum and sum-of-squares,
// thread 0 combines both in shared memory, and every thread normalizes its
// own slice.
import common_norm;

StructuredBuffer<float> g_Src;
StructuredBuffer<float> g_Alpha;
StructuredBuffer<float> g_Beta;
RWStructuredBuffer<float> g_Dst;
layout(push_constant) Params g_Params;

static groupshared float gRedSum[256];
static groupshared float gRedSq[256];

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
    float p2 = 0.0f;
    for (uint64_t j = tid; j < n; j += 256u)
    {
        float v = g_Src[base + j];
        ps += v;
        p2 += v * v;
    }
    gRedSum[tid] = ps;
    gRedSq[tid] = p2;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        for (uint t = 1u; t < 256u; ++t)
        {
            gRedSum[0] = gRedSum[0] + gRedSum[t];
            gRedSq[0] = gRedSq[0] + gRedSq[t];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    double mean = double(gRedSum[0]) / double(n);
    // `var` uses E[x^2] - E[x]^2, matching the candle CPU `LayerNorm` reference
    // exactly. This form can lose precision (catastrophic cancellation) when the
    // mean is large relative to the variance, but switching to the two-pass
    // stable form here would make the GPU diverge from the CPU reference, so
    // keep the parity until the CPU reference itself changes.
    double var = double(gRedSq[0]) / double(n) - mean * mean;
    double inv_std = 1.0 / sqrt(var + g_Params.eps);
    float mean_f = float(mean);
    float inv_std_f = float(inv_std);
    for (uint64_t j = tid; j < n; j += 256u)
    {
        float x = g_Src[base + j];
        g_Dst[base + j] = (x - mean_f) * inv_std_f * g_Alpha[j] + g_Beta[j];
    }
}