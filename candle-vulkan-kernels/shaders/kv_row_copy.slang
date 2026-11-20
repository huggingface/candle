// kv_row_copy.slang
// Fused KV cache row append: one new key row and one new value row into
// pre-allocated f16 (heads, max_len, head_dim) caches, in a single
// dispatch (two dispatches otherwise, one per cache). The src rows are
// f32 activations; the copy converts to f16 on store.
//
// src_k/src_v hold (heads * head_dim) contiguous f32 values (head-major);
// the row is written to dst[head * stride + row * head_dim + col] of
// dst_k and dst_v.

import common_kv_row_copy;

StructuredBuffer<float> g_SrcK;
StructuredBuffer<float> g_SrcV;
RWStructuredBuffer<half> g_DstK;
RWStructuredBuffer<half> g_DstV;
layout(push_constant) Params g_Params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint64_t half_count = uint64_t(g_Params.heads) * g_Params.head_dim;
    uint64_t e = dispatchThreadID.x;
    if (e >= 2 * half_count)
    {
        return;
    }
    bool is_v = e >= half_count;
    uint64_t i = is_v ? e - half_count : e;
    uint64_t head = i / g_Params.head_dim;
    uint64_t col = i % g_Params.head_dim;
    uint64_t dst = head * g_Params.stride + g_Params.row * g_Params.head_dim + col;
    if (is_v)
    {
        g_DstV[dst] = (half)g_SrcV[i];
    }
    else
    {
        g_DstK[dst] = (half)g_SrcK[i];
    }
}
