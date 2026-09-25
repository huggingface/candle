// Indexed copy: out[i] = src[idx[i / blen] + (i % blen)].
// blen == 1 -> element-wise gather; blen > 1 -> block gather.
StructuredBuffer<float> g_Src;
RWStructuredBuffer<float> g_Dst;
StructuredBuffer<uint> g_Idx;
StructuredBuffer<float> g_Params; // [0]=total, [1]=block_len

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i >= uint(g_Params[0]))
    {
        return;
    }
    uint blen = max(uint(g_Params[1]), 1u);
    uint k = i / blen;
    uint j = i - k * blen;
    g_Dst[i] = g_Src[g_Idx[k] + j];
}
