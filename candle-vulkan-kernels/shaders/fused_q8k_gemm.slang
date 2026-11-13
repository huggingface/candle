// Fused Q8_K decode GEMM: out[n] = lhs[k] @ dequant(W[n, k]).
//
// Single-row lhs (decode): each thread owns one output column and
// dequantizes its Q8_K column in registers while accumulating the dot
// product. W holds raw Q8_K blocks (256 elements per 292 bytes: 4-byte f32 scale, then 256 i8 quants) stored
// (n, k)-contiguous. m must be 1 and k a multiple of 256; n need not be a
// multiple of 32 (out-of-range columns only skip the reduction and store,
// but still help stage the activation and hit the barriers).
//
// The activation g_Lhs is column-independent: without staging it is read
// from global/L2 once per output column. Each 256-element activation
// block is therefore staged once in groupshared memory per workgroup, so
// the 32 threads read it from LDS instead of re-reading it n times. Four
// independent accumulators break the serial FMA chain (ILP).
//
// Barrier discipline: GroupMemoryBarrierWithGroupSync() is UB in
// divergent control flow (Vulkan spec 15.5.3), and RADV miscompiles it
// there. Every thread therefore executes both barriers on every block
// iteration, including the out-of-range column threads.
//
// Params: [0]=m (must be 1), [1]=n, [2]=k.
import dequant_helpers;
import common_fused_gemm;

StructuredBuffer<float> g_Lhs; // (k,) f32
RWStructuredBuffer<float> g_Out; // (n,) f32
StructuredBuffer<uint8_t> g_W; // raw Q8_K bytes, (n, k)
layout(push_constant) Params g_Params;
static groupshared float lhs_l2s[256];

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    uint64_t n = g_Params.n;
    uint64_t k = g_Params.k;
    uint col = groupID.y * 32u + dispatchThreadID.x;
    bool in_range = col < n;
    uint kblocks = k / 256u;
    // Four independent accumulators break the serial FMA chain (ILP).
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        // Stage one 256-element activation block in LDS (coalesced).
        uint lbase = bb * 256u;
        for (uint j = 0u; j < 8u; ++j)
        {
            uint gidx = lbase + localID.x * 8u + j;
            lhs_l2s[localID.x * 8u + j] = (gidx < k) ? g_Lhs[gidx] : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();
        if (in_range)
        {
            uint off = (col * kblocks + bb) * 292u;
            uint32_t dbits = uint32_t(g_W[off]) | (uint32_t(g_W[off + 1u]) << 8)
                | (uint32_t(g_W[off + 2u]) << 16) | (uint32_t(g_W[off + 3u]) << 24);
            float d = asfloat(dbits);
            for (uint j = 0u; j < 256u; ++j)
            {
                int q = sbyte(g_W[off + 4u + j]);
                float val = lhs_l2s[j] * (float(q) * d);
                uint r = j & 3u;
                if (r == 0u) { a0 += val; }
                else if (r == 1u) { a1 += val; }
                else if (r == 2u) { a2 += val; }
                else { a3 += val; }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (in_range)
    {
        g_Out[col] = a0 + a1 + a2 + a3;
    }
}