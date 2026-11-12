// Last-axis reductions, one 256-thread workgroup per row:
// out[row] = sum_c in[row*cols + c]                      (main_reduce_sum)
// out[row] = max_c in[row*cols + c]                      (main_reduce_max)
// out[row] = min_c in[row*cols + c]                      (main_reduce_min)
// out[row] = argmin_c in[row*cols + c]                   (main_reduce_argmin)
// out[row] = argmax_c in[row*cols + c]                   (main_reduce_argmax)
//
// Each thread folds its strided slice of the row (for argmin/argmax the
// earliest index of the best value in the slice, matching the serial
// kernels' strict comparison), then thread 0 combines the 256 partials in a
// serial loop over shared memory — the same uniform-barrier pattern as
// softmax/decode_attention, since a divergent tree reduction reaches
// barriers from divergent control flow and RADV miscompiles it. Threads with
// an empty slice (cols < 256) contribute the reduction identity.
import common_reduce;
StructuredBuffer<float> g_In;
RWStructuredBuffer<float> g_OutF;
RWStructuredBuffer<uint> g_OutI;
layout(push_constant) Params g_Params;

static groupshared float gRedF[256];
static groupshared uint gRedI[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main_reduce_min(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint row = gid.x;
    uint tid = gtid.x;
    uint64_t rows = g_Params.rows;
    uint64_t cols = g_Params.cols;
    if (row >= rows || cols == 0u)
    {
        return;
    }
    uint64_t base = row * cols;
    float pm = 3.402823466e+38f;
    for (uint c = tid; c < cols; c += 256u)
    {
        pm = min(pm, g_In[base + c]);
    }
    gRedF[tid] = pm;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        for (uint t = 1u; t < 256u; ++t)
        {
            gRedF[0] = min(gRedF[0], gRedF[t]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    g_OutF[row] = gRedF[0];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_reduce_argmin(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint row = gid.x;
    uint tid = gtid.x;
    uint64_t rows = g_Params.rows;
    uint64_t cols = g_Params.cols;
    if (row >= rows || cols == 0u)
    {
        return;
    }
    uint64_t base = row * cols;
    float bestv = 3.402823466e+38f;
    uint besti = 0u;
    for (uint c = tid; c < cols; c += 256u)
    {
        float v = g_In[base + c];
        if (v < bestv)
        {
            bestv = v;
            besti = c;
        }
    }
    gRedF[tid] = bestv;
    gRedI[tid] = besti;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        float bv = gRedF[0];
        uint bi = gRedI[0];
        for (uint t = 1u; t < 256u; ++t)
        {
            if (gRedF[t] < bv)
            {
                bv = gRedF[t];
                bi = gRedI[t];
            }
        }
        gRedF[0] = bv;
        gRedI[0] = bi;
    }
    GroupMemoryBarrierWithGroupSync();
    g_OutI[row] = gRedI[0];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_reduce_argmax(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint row = gid.x;
    uint tid = gtid.x;
    uint64_t rows = g_Params.rows;
    uint64_t cols = g_Params.cols;
    if (row >= rows || cols == 0u)
    {
        return;
    }
    uint64_t base = row * cols;
    float bestv = -3.402823466e+38f;
    uint besti = 0u;
    for (uint c = tid; c < cols; c += 256u)
    {
        float v = g_In[base + c];
        if (v > bestv)
        {
            bestv = v;
            besti = c;
        }
    }
    gRedF[tid] = bestv;
    gRedI[tid] = besti;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        float bv = gRedF[0];
        uint bi = gRedI[0];
        for (uint t = 1u; t < 256u; ++t)
        {
            if (gRedF[t] > bv)
            {
                bv = gRedF[t];
                bi = gRedI[t];
            }
        }
        gRedF[0] = bv;
        gRedI[0] = bi;
    }
    GroupMemoryBarrierWithGroupSync();
    g_OutI[row] = gRedI[0];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_reduce_sum(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint row = gid.x;
    uint tid = gtid.x;
    uint64_t rows = g_Params.rows;
    uint64_t cols = g_Params.cols;
    if (row >= rows || cols == 0u)
    {
        return;
    }
    uint64_t base = row * cols;
    float ps = 0.0f;
    for (uint c = tid; c < cols; c += 256u)
    {
        ps += g_In[base + c];
    }
    gRedF[tid] = ps;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        for (uint t = 1u; t < 256u; ++t)
        {
            gRedF[0] = gRedF[0] + gRedF[t];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    g_OutF[row] = gRedF[0];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main_reduce_max(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint row = gid.x;
    uint tid = gtid.x;
    uint64_t rows = g_Params.rows;
    uint64_t cols = g_Params.cols;
    if (row >= rows || cols == 0u)
    {
        return;
    }
    uint64_t base = row * cols;
    float pm = -3.402823466e+38f;
    for (uint c = tid; c < cols; c += 256u)
    {
        pm = max(pm, g_In[base + c]);
    }
    gRedF[tid] = pm;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        for (uint t = 1u; t < 256u; ++t)
        {
            gRedF[0] = max(gRedF[0], gRedF[t]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    g_OutF[row] = gRedF[0];
}