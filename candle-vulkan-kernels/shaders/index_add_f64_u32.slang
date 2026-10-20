// index_add: g_Dst[(pre_i * max_idx + ids[src_idx]) * post + r] += g_Src[i],
// executed atomically (SPV_EXT_shader_atomic_float_add). For src element i:
// i = (pre_i * nsrc + src_idx) * post + r. Skipped when ids[src_idx] is the
// ids-dtype max value (candle's skip sentinel) or out of self's range.
RWStructuredBuffer<double> g_Dst;
StructuredBuffer<double> g_Src;
StructuredBuffer<uint> g_Ids;
StructuredBuffer<uint64_t> g_Params; // [0]=total, [1]=post_dim, [2]=nsrc, [3]=max_idx
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    uint64_t total = g_Params[0];
    uint64_t post = max(g_Params[1], 1u);
    uint64_t nsrc = max(g_Params[2], 1u);
    uint64_t maxidx = g_Params[3];
    if (i >= total) { return; }
    uint64_t src_idx = (i / post) % nsrc;
    uint64_t pre_i = i / (nsrc * post);
    uint64_t r = i % post;
    uint id = g_Ids[src_idx];
    if (id == 0xFFFFFFFFu || id >= maxidx) { return; }
    uint64_t uidx = id;
    uint64_t dst = (pre_i * maxidx + uidx) * post + r;
    InterlockedAdd(g_Dst[dst], g_Src[i]);
}
