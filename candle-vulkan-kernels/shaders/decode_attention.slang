// decode_attention.slang
// Fused KV-cache decode attention (bf16), one workgroup per query head.
//
// out[h, :] = softmax( (Q[h, :] @ K[g, :].T) * scale ) @ V[g, :]
//   where g = h * num_kv_heads / num_heads (GQA head -> kv-head group).
//
// Q: [num_heads, head_dim], K: [num_kv_heads, kv_len, head_dim],
// V: [num_kv_heads, kv_len, head_dim] (all contiguous F32).
//
// The 32 threads of the workgroup cooperate to compute the scores over the
// kv_len dimension (strided), then a block-reduce (max + sum) for the
// numerically-stable softmax, then the output matvec (strided over head_dim).
//
// NOTE: heads are assigned from groupID.y (the dispatch is [1, num_heads, 1]).
// RADV miscompiles multi-workgroup dispatches along the X axis for this kernel
// (only the x=0 workgroup runs), so the workgroups are laid out along Y.
//
// All strided loops are UNIFORM in iteration count (every thread executes the
// same number of iterations, with an in-loop guard), because RADV miscompiles
// barriers reached from divergent control flow (Vulkan spec 15.5.3) -- the
// same reason gemm.slang runs both barriers on every slab unconditionally.
// Workgroup shared memory lives at global scope, as in gemm.slang.

struct Params
{
    uint64_t num_heads;
    uint64_t num_kv_heads;
    uint64_t head_dim;
    uint64_t kv_len;
    float scale;
};
layout(push_constant) Params g_Params;

StructuredBuffer<float> g_Q;
StructuredBuffer<float> g_K;
StructuredBuffer<float> g_V;
RWStructuredBuffer<float> g_Out;

static const uint MAX_KV = 4096u;

// Workgroup shared memory (global scope, as in gemm.slang).
static groupshared float scores[MAX_KV];
static groupshared float red[32];

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID)
{
    uint h = groupID.y;
    uint D = uint(g_Params.head_dim);
    if (h >= uint(g_Params.num_heads))
    {
        return;
    }
    uint kv_len = uint(g_Params.kv_len);
    uint g = h * uint(g_Params.num_kv_heads) / uint(g_Params.num_heads);
    float scale = g_Params.scale;
    uint tid = dispatchThreadID.x;

    // scores[j] = (q[h] . K[g, j]) * scale. Uniform loop, in-loop guard.
    uint qbase = h * D;
    for (uint j = 0u; j < kv_len; j += 32u)
    {
        uint jj = j + tid;
        if (jj < kv_len)
        {
            float s = 0.0;
            uint kbase = (g * kv_len + jj) * D;
            for (uint d = 0u; d < D; ++d)
            {
                s += g_Q[qbase + d] * g_K[kbase + d];
            }
            scores[jj] = s * scale;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Block max over the scores.
    float local_max = -1e30f;
    for (uint j = 0u; j < kv_len; j += 32u)
    {
        uint jj = j + tid;
        if (jj < kv_len)
        {
            local_max = max(local_max, scores[jj]);
        }
    }
    red[tid] = local_max;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        for (uint i = 1u; i < 32u; ++i)
        {
            red[0] = max(red[0], red[i]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float m = red[0];

    // In-place exps plus block sum.
    float local_sum = 0.0;
    for (uint j = 0u; j < kv_len; j += 32u)
    {
        uint jj = j + tid;
        if (jj < kv_len)
        {
            scores[jj] = exp(scores[jj] - m);
            local_sum += scores[jj];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    red[tid] = local_sum;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        for (uint i = 1u; i < 32u; ++i)
        {
            red[0] += red[i];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float S = red[0];

    // out[h, d] = (sum_j scores[j] * V[g, j, d]) / S.
    uint vbase = g * kv_len * D;
    for (uint d = 0u; d < D; d += 32u)
    {
        uint dd = d + tid;
        if (dd < D)
        {
            float acc = 0.0;
            for (uint j = 0u; j < kv_len; ++j)
            {
                acc += scores[j] * g_V[vbase + j * D + dd];
            }
            g_Out[h * D + dd] = acc / S;
        }
    }
}
