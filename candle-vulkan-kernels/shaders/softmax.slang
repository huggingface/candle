// Softmax over the last dim, one 256-thread workgroup per row:
// out[i][j] = exp(x[i][j] - max) / sum_k(exp(x[i][k] - max)).
//
// Each thread computes a partial max and a partial sum over its strided
// slice of the row, writing the exponentiated values to the output as it
// goes. The partials are combined by thread 0 in a serial loop over the
// shared-memory array (the same uniform pattern as decode_attention): a
// tree reduction's `if (tid < off)` would reach barriers from divergent
// control flow, which RADV miscompiles. Threads with an empty slice (when
// n < 256) contribute the reduction identity (-FLT_MAX for max, 0 for the
// sum) and still reach every barrier, keeping the barriers uniform.
import common_softmax;
StructuredBuffer<float> g_Src;
RWStructuredBuffer<float> g_Dst;
layout(push_constant) Params g_Params;

static groupshared float gRed[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    uint i = groupID.x;
    uint tid = groupThreadID.x;
    uint64_t n_rows = g_Params.n_rows;
    if (i >= n_rows)
    {
        return;
    }
    uint64_t n = g_Params.n;
    uint64_t base = i * n;

    // Partial max over this thread's strided slice.
    float pm = -3.402823466e+38f;
    for (uint j = tid; j < n; j += 256u)
    {
        pm = max(pm, g_Src[base + j]);
    }
    gRed[tid] = pm;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        for (uint t = 1u; t < 256u; ++t)
        {
            gRed[0] = max(gRed[0], gRed[t]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float mx = gRed[0];

    // Partial sum of exp, writing the exponentiated values to g_Dst.
    float ps = 0.0f;
    for (uint j = tid; j < n; j += 256u)
    {
        float e = exp(g_Src[base + j] - mx);
        g_Dst[base + j] = e;
        ps += e;
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
    float inv = 1.0f / gRed[0];

    // Normalize each thread's slice.
    for (uint j = tid; j < n; j += 256u)
    {
        g_Dst[base + j] = g_Dst[base + j] * inv;
    }
}