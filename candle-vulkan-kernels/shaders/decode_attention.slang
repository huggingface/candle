// decode_attention.slang
// Two-pass flash-decode single-query decode attention (f32).
//
// out[h, :] = softmax( (Q[h, :] @ K[g, :].T) * scale ) @ V[g, :]
//   where g = h * num_kv_heads / num_heads (GQA head -> kv-head group).
//
// Q: [num_heads, head_dim], K: [num_kv_heads, kv_len, head_dim],
// V: [num_kv_heads, kv_len, head_dim] (all F32). K and V rows are
// contiguous (stride head_dim); consecutive heads are `kv_stride`
// elements apart (kv_stride >= kv_len * head_dim; equal for a
// fully contiguous cache, larger for a pre-allocated one).
//
// Pass 1 (`main`): the kv_len dimension is split into chunks of CHUNK
// positions (one workgroup per (head, chunk), dispatched along Y and
// Z). Each lane streams its own strided subset of the chunk's
// positions (4 slots of 32), computing each position's full head_dim
// score (no per-position barrier) and storing the slot scores in
// workgroup memory. After a block max and a block sum, each lane folds
// every position's V row into its own head_dim/32-wide accumulator
// slice (the V reads are coalesced across the workgroup). The block
// writes one partial (m, S, acc[head_dim]) per (head, chunk) into
// `g_Scratch`, laid out as [num_heads][num_chunks][2 + head_dim].
//
// Pass 2 (`decode_attention_combine`): one 32-thread workgroup per head
// merges that head's num_chunks partials with a second online softmax
// (each lane owns its head_dim/32 output slice) and writes the final
// (head_dim) row.
//
// No groupshared scores buffer is involved, so kv_len is unbounded (the
// old MAX_KV = 4096 cap is gone). head_dim must be <= MAX_D.
//
// NOTE: heads are assigned from groupID.y (the dispatch is
// [1, num_heads, num_chunks]). RADV miscompiles multi-workgroup
// dispatches along the X axis for this kernel (only the x=0 workgroup
// runs), so the workgroups are laid out along Y and Z.
//
// All strided loops are UNIFORM in iteration count (every thread executes
// the same number of iterations, with an in-loop guard), because RADV
// miscompiles barriers reached from divergent control flow (Vulkan spec
// 15.5.3) -- the same reason gemm.slang runs both barriers on every slab
// unconditionally. Workgroup shared memory lives at global scope, as in
// gemm.slang.

struct Params
{
    uint64_t num_heads;
    uint64_t num_kv_heads;
    uint64_t head_dim;
    uint64_t kv_len;
    uint64_t kv_stride;
    float scale;
};
layout(push_constant) Params g_Params;

// Bindings are pinned explicitly: Slang auto-assignment is per entry
// point, and the two passes reference different subsets of the
// globals (pass 1: q/k/v/scratch, pass 2: scratch/out).
[[vk::binding(0)]] StructuredBuffer<float> g_Q;
[[vk::binding(1)]] StructuredBuffer<float> g_K;
[[vk::binding(2)]] StructuredBuffer<float> g_V;
[[vk::binding(3)]] RWStructuredBuffer<float> g_Scratch;
[[vk::binding(4)]] RWStructuredBuffer<float> g_Out;

// Positions per (head, chunk) workgroup; must match
// `DECODE_ATTENTION_CHUNK` in the Rust caller.
static const uint CHUNK = 128u;
// Largest supported head_dim: the per-lane accumulator is
// head_dim / 32 floats, capped at 16.
static const uint MAX_D = 512u;

// Workgroup shared memory (global scope, as in gemm.slang).
static groupshared float red[32];
// Per-lane slot scores (32 lanes x 4 slots), written by pass 1.
static groupshared float g_s[128];

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID)
{
    uint h = groupID.y;
    uint c = groupID.z;
    uint D = uint(g_Params.head_dim);
    if (h >= uint(g_Params.num_heads))
    {
        return;
    }
    uint kv_len = uint(g_Params.kv_len);
    uint kv_stride = uint(g_Params.kv_stride);
    uint num_chunks = (kv_len + CHUNK - 1u) / CHUNK;
    if (c >= num_chunks)
    {
        return;
    }
    uint g = h * uint(g_Params.num_kv_heads) / uint(g_Params.num_heads);
    float scale = g_Params.scale;
    uint tid = dispatchThreadID.x;
    uint chunk = (D + 31u) / 32u;
    uint d0 = tid * chunk;

    uint j0 = c * CHUNK;
    uint jend = min(kv_len, j0 + CHUNK);
    uint qbase = h * D;
    uint khead = g * kv_stride;

    // This lane's positions: j = j0 + tid + 32*slot (4 slots per lane,
    // CHUNK = 128 = 32 lanes x 4 slots). Each lane computes the full
    // head_dim score of its positions and stores them in workgroup
    // memory; invalid slots stay at -FLT_MAX so the block reductions
    // below are uniform.
    for (uint slot = 0u; slot < 4u; ++slot)
    {
        uint j = j0 + tid + 32u * slot;
        float s = -3.402823466e+38f;
        if (j < jend)
        {
            uint base = khead + j * D;
            float dot = 0.0f;
            for (uint d = 0u; d < D; ++d)
            {
                dot += g_Q[qbase + d] * g_K[base + d];
            }
            s = dot * scale;
        }
        g_s[tid * 4u + slot] = s;
    }
    GroupMemoryBarrierWithGroupSync();

    // Block max over all (lane, slot) scores -> m_c.
    float lm = -3.402823466e+38f;
    for (uint slot = 0u; slot < 4u; ++slot)
    {
        lm = max(lm, g_s[tid * 4u + slot]);
    }
    red[tid] = lm;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        for (uint i = 1u; i < 32u; ++i)
        {
            red[0] = max(red[0], red[i]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float m_c = red[0];

    // Block sum of exp(s - m_c) -> S_c.
    float ls = 0.0f;
    for (uint slot = 0u; slot < 4u; ++slot)
    {
        ls += exp(g_s[tid * 4u + slot] - m_c);
    }
    red[tid] = ls;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        for (uint i = 1u; i < 32u; ++i)
        {
            red[0] += red[i];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float S_c = red[0];

    // Fold every position's V row into this lane's d-slice. For a fixed
    // (lane, slot), all 32 lanes read one D-wide V row, so the reads
    // are coalesced. Every output d is owned by exactly one lane
    // (chunk = ceil(D/32)), so no cross-lane reduction of the
    // accumulator is needed.
    float acc[16];
    for (uint i = 0u; i < chunk; ++i)
    {
        acc[i] = 0.0f;
    }
    for (uint t = 0u; t < 32u; ++t)
    {
        for (uint slot = 0u; slot < 4u; ++slot)
        {
            uint j = j0 + t + 32u * slot;
            if (j < jend)
            {
                float w = exp(g_s[t * 4u + slot] - m_c);
                uint vbase = khead + j * D;
                for (uint i = 0u; i < chunk; ++i)
                {
                    if (d0 + i < D)
                    {
                        acc[i] += g_V[vbase + d0 + i] * w;
                    }
                }
            }
        }
    }

    uint pbase = (h * num_chunks + c) * (2u + D);
    g_Scratch[pbase] = m_c;
    g_Scratch[pbase + 1u] = S_c;
    for (uint i = 0u; i < chunk; ++i)
    {
        if (d0 + i < D)
        {
            g_Scratch[pbase + 2u + d0 + i] = acc[i];
        }
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void decode_attention_combine(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID)
{
    uint h = groupID.y;
    uint D = uint(g_Params.head_dim);
    if (h >= uint(g_Params.num_heads))
    {
        return;
    }
    uint kv_len = uint(g_Params.kv_len);
    uint num_chunks = (kv_len + CHUNK - 1u) / CHUNK;
    uint tid = dispatchThreadID.x;
    uint chunk = (D + 31u) / 32u;
    uint d0 = tid * chunk;

    // Merge the chunk partials of head h with an online softmax over the
    // partials (each partial is a (m, S, acc) state, so combining is the
    // same max/rescale/accumulate step as in pass 1).
    float m_g = -3.402823466e+38f;
    for (uint c = 0u; c < num_chunks; ++c)
    {
        m_g = max(m_g, g_Scratch[(h * num_chunks + c) * (2u + D)]);
    }
    float S_g = 0.0f;
    for (uint c = 0u; c < num_chunks; ++c)
    {
        uint pbase = (h * num_chunks + c) * (2u + D);
        S_g += g_Scratch[pbase + 1u] * exp(g_Scratch[pbase] - m_g);
    }
    float inv = (S_g > 0.0f) ? 1.0f / S_g : 0.0f;

    float acc[16];
    for (uint i = 0u; i < chunk; ++i)
    {
        acc[i] = 0.0f;
    }
    for (uint c = 0u; c < num_chunks; ++c)
    {
        uint pbase = (h * num_chunks + c) * (2u + D);
        float a = exp(g_Scratch[pbase] - m_g);
        uint off = d0;
        for (uint i = 0u; i < chunk; ++i)
        {
            if (off + i < D)
            {
                acc[i] += g_Scratch[pbase + 2u + off + i] * a;
            }
        }
    }

    uint off = d0;
    for (uint i = 0u; i < chunk; ++i)
    {
        if (off + i < D)
        {
            g_Out[h * D + off + i] = acc[i] * inv;
        }
    }
}
