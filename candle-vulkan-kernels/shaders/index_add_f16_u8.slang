// index_add: g_Dst[(pre_i * max_idx + ids[src_idx]) * post + r] += g_Src[i],
// executed atomically (SPV_EXT_shader_atomic_float_add). For src element i:
// i = (pre_i * nsrc + src_idx) * post + r. Skipped when ids[src_idx] is the
// ids-dtype max value (candle's skip sentinel) or out of self's range.
import common_index_add;
RWStructuredBuffer<half> g_Dst;
StructuredBuffer<half> g_Src;
StructuredBuffer<uint8_t> g_Ids;
layout(push_constant) Params g_Params;
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    uint64_t total = g_Params.count;
    uint64_t post = max(g_Params.post_dim, 1u);
    uint64_t nsrc = max(g_Params.nsrc, 1u);
    uint64_t maxidx = g_Params.max_idx;
    if (i >= total) { return; }
    uint64_t src_idx = (i / post) % nsrc;
    uint64_t pre_i = i / (nsrc * post);
    uint64_t r = i % post;
    uint id = (uint)g_Ids[src_idx];
    if (id == 255u || id >= maxidx) { return; }
    uint64_t uidx = id;
    uint64_t dst = (pre_i * maxidx + uidx) * post + r;
    InterlockedAdd(g_Dst[dst], g_Src[i]);
}
