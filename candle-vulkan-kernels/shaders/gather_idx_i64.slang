// i64 indexed copy.
// Indexed copy: out[i] = src[idx[i / blen] + (i % blen)].
// blen == 1 -> element-wise gather; blen > 1 -> block gather.
StructuredBuffer<int64_t> g_Src;
RWStructuredBuffer<int64_t> g_Dst;
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
