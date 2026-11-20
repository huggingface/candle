// i64 indexed copy.
// Indexed copy: out[i] = src[idx[i / blen] + (i % blen)].
// blen == 1 -> element-wise gather; blen > 1 -> block gather.
import common_gather_idx;
StructuredBuffer<int64_t> g_Src;
RWStructuredBuffer<int64_t> g_Dst;
StructuredBuffer<uint> g_Idx;
layout(push_constant) Params g_Params;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint64_t i = dispatchThreadID.x;
    if (i >= g_Params.count)
    {
        return;
    }
    uint64_t blen = max(g_Params.block_len, 1u);
    uint64_t k = i / blen;
    uint64_t j = i - k * blen;
    g_Dst[i + g_Params.emb_h] = g_Src[g_Idx[k] + j];
}
void main_index_select(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint64_t total = g_Params.count;
    uint64_t inner = max(g_Params.block_len, 1u);
    uint64_t emb_h = max(g_Params.emb_h, 1u);
    if (o >= total)
    {
        return;
    }
    uint64_t j = o % inner;
    uint64_t i = o / inner;
    g_Dst[o + g_Params.dst_offset] = g_Src[g_Idx[i] * emb_h + j];
}
