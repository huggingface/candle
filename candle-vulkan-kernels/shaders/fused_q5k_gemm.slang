// Fused Q5_K decode GEMM: out[n] = lhs[k] @ dequant(W[n, k]).
//
// Single-row lhs (decode): each thread owns one output column and
// dequantizes its Q5_K column in registers while accumulating the dot
// product. W holds raw Q5_K blocks (256 elements per 176 bytes: 2 f16 super scale/min, 12 bytes of 6-bit scale/min pairs, 32 bytes of high bits, 128 bytes of 4-bit quants) stored
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
StructuredBuffer<uint8_t> g_W; // raw Q5_K bytes, (n, k)
layout(push_constant) Params g_Params;
static groupshared float lhs_l2s[256];

// Q4K/Q5K scale-min decoder (from candle quantized/utils.rs::get_scale_min_k4).
// `off` is the block base offset in g_W; `soff` is the scale-array offset.
uint2 get_scale_min_k4(uint j, uint off, uint soff)
{
    if (j < 4u)
    {
        uint d = g_W[off + soff + j] & 63u;
        uint m = g_W[off + soff + j + 4u] & 63u;
        return uint2(d, m);
    }
    else
    {
        uint d = (g_W[off + soff + j + 4u] & 0xFu) | ((g_W[off + soff + j - 4u] >> 6) << 4);
        uint m = (g_W[off + soff + j + 4u] >> 4) | ((g_W[off + soff + j] >> 6) << 4);
        return uint2(d, m);
    }
}

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
            uint off = (col * kblocks + bb) * 176u;
            float d =
                f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
            float mn =
                f16_bits_to_f32(uint16_t(g_W[off + 2u]) | (uint16_t(g_W[off + 3u]) << 8u));
            for (uint s = 0u; s < 8u; ++s)
            {
                uint2 sm = get_scale_min_k4(s, off, 4u);
                float d1 = d * float(sm.x);
                float m1 = mn * float(sm.y);
                uint ql0 = off + 48u + 32u * (s / 2u);
                uint k0 = s * 32u;
                uint bit = 1u << s;
                if ((s & 1u) == 0u)
                {
                    for (uint i = 0u; i < 32u; ++i)
                    {
                        uint qv = g_W[ql0 + i] & 0xFu;
                        uint to_add = (g_W[off + 16u + i] & bit) != 0u ? 16u : 0u;
                        float val = lhs_l2s[k0 + i] * (d1 * float(qv + to_add) - m1);
                        uint r = i & 3u;
                        if (r == 0u) { a0 += val; }
                        else if (r == 1u) { a1 += val; }
                        else if (r == 2u) { a2 += val; }
                        else { a3 += val; }
                    }
                }
                else
                {
                    for (uint i = 0u; i < 32u; ++i)
                    {
                        uint qv = (g_W[ql0 + i] >> 4u) & 0xFu;
                        uint to_add = (g_W[off + 16u + i] & bit) != 0u ? 16u : 0u;
                        float val = lhs_l2s[k0 + i] * (d1 * float(qv + to_add) - m1);
                        uint r = i & 3u;
                        if (r == 0u) { a0 += val; }
                        else if (r == 1u) { a1 += val; }
                        else if (r == 2u) { a2 += val; }
                        else { a3 += val; }
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (in_range)
    {
        g_Out[col] = a0 + a1 + a2 + a3;
    }
}