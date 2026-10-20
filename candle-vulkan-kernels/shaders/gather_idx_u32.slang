// u32 indexed copy.
// Indexed copy: out[i] = src[idx[i / blen] + (i % blen)].
// blen == 1 -> element-wise gather; blen > 1 -> block gather.
StructuredBuffer<uint> g_Src;
RWStructuredBuffer<uint> g_Dst;
StructuredBuffer<uint> g_Idx;
StructuredBuffer<uint64_t> g_Params; // [0]=total, [1]=block_len, [2]=dst_offset
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint64_t i = dispatchThreadID.x;
    if (i >= g_Params[0])
    {
        return;
    }
    uint64_t blen = max(g_Params[1], 1u);
    uint64_t k = i / blen;
    uint64_t j = i - k * blen;
    g_Dst[i + g_Params[2]] = g_Src[g_Idx[k] + j];
}
// Row gather: out[o] = src[idx[o] * emb_h + (o % inner)], with
// o = i * inner + j. `inner` is the number of ids columns, `emb_h`
// the number of embedding columns.
// Params (this entry): [0]=total, [1]=inner, [2]=emb_h, [3]=dst_offset.
[shader("compute")]
[numthreads(256, 1, 1)]
void main_gather_rows(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint64_t total = g_Params[0];
    uint64_t inner = max(g_Params[1], 1u);
    uint64_t emb_h = max(g_Params[2], 1u);
    if (o >= total)
    {
        return;
    }
    uint64_t j = o % inner;
    g_Dst[o + g_Params[3]] = g_Src[g_Idx[o] * emb_h + j];
}
// Slice select: out[o] = src[idx[o / inner] * emb_h + (o % inner)],
// o = i * inner + j (ids are per-slice, not per-element).
// Params (this entry): [0]=total, [1]=inner, [2]=emb_h, [3]=dst_offset.
[shader("compute")]
[numthreads(256, 1, 1)]
void main_index_select(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint o = dispatchThreadID.x;
    uint64_t total = g_Params[0];
    uint64_t inner = max(g_Params[1], 1u);
    uint64_t emb_h = max(g_Params[2], 1u);
    if (o >= total)
    {
        return;
    }
    uint64_t j = o % inner;
    uint64_t i = o / inner;
    g_Dst[o + g_Params[3]] = g_Src[g_Idx[i] * emb_h + j];
}
