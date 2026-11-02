// Fused Q4_K routed MoE GEMM:
// out[route, n] = lhs[route, k] @ dequant(W[ids[route], n, k]).
//
// The routed quantized MoE projection for the decode/prefill path: for
// each route (a flattened (token, topk) pair), compute the dequantized
// GEMV of the route's input row against the selected expert's weights,
// reading the quantized bytes on the fly (no f32 materialization of the
// expert weights). Each thread owns one (route, output column); the
// dequantization happens in registers, so no shared memory or barriers
// are needed. W holds the raw Q4_K bytes of all experts, (experts, n,
// k)-contiguous; `weight_expert_bytes` is the byte size of one expert's
// (n, k) slice. k must be a multiple of 256; n need not be a multiple of
// 32 (out-of-range columns early-exit).
//
// Params: [0]=n, [1]=k, [2]=weight_expert_bytes.
import dequant_helpers;
import common_moe_gemm;

StructuredBuffer<float> g_Lhs; // (num_routes, k) f32
RWStructuredBuffer<float> g_Out; // (num_routes, n) f32
StructuredBuffer<uint8_t> g_W; // raw Q4_K bytes, (experts, n, k)
StructuredBuffer<uint> g_Ids; // (num_routes,) u32 expert ids
layout(push_constant) Params g_Params;

// 6-bit scale/min pair index j (0..7) of the Q4_K block at `off`
// (from candle quantized/utils.rs::get_scale_min_k4, soff=4).
uint2 get_scale_min_k4(uint j, uint off)
{
    if (j < 4u)
    {
        uint d = g_W[off + 4u + j] & 63u;
        uint m = g_W[off + 8u + j] & 63u;
        return uint2(d, m);
    }
    else
    {
        uint d = (g_W[off + 12u + (j - 4u)] & 0xFu) | ((g_W[off + 4u + (j - 4u)] >> 6) << 4);
        uint m = (g_W[off + 12u + (j - 4u)] >> 4) | ((g_W[off + 8u + (j - 4u)] >> 6) << 4);
        return uint2(d, m);
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID)
{
    uint64_t n = g_Params.n;
    uint64_t k = g_Params.k;
    uint64_t wexp = g_Params.weight_expert_bytes;
    uint col = groupID.y * 32u + dispatchThreadID.x;
    if (col >= n)
    {
        return;
    }
    uint route = groupID.z;
    uint expert = g_Ids[route];
    uint wbase = expert * (uint)wexp;
    uint kblocks = (uint)k / 256u;
    uint lhs_base = route * (uint)k;
    float acc = 0.0;
    for (uint bb = 0u; bb < kblocks; ++bb)
    {
        uint off = wbase + (col * kblocks + bb) * 144u;
        float d =
            f16_bits_to_f32(uint16_t(g_W[off]) | (uint16_t(g_W[off + 1u]) << 8u));
        float mn =
            f16_bits_to_f32(uint16_t(g_W[off + 2u]) | (uint16_t(g_W[off + 3u]) << 8u));
        for (uint s = 0u; s < 8u; ++s)
        {
            uint2 sm = get_scale_min_k4(s, off);
            float d1 = d * float(sm.x);
            float m1 = mn * float(sm.y);
            uint qbyte0 = off + 16u + 32u * (s / 2u);
            uint k0 = (bb * 8u + s) * 32u;
            if ((s & 1u) == 0u)
            {
                for (uint i = 0u; i < 32u; ++i)
                {
                    acc += g_Lhs[lhs_base + k0 + i] * (d1 * float(g_W[qbyte0 + i] & 0xFu) - m1);
                }
            }
            else
            {
                for (uint i = 0u; i < 32u; ++i)
                {
                    acc +=
                        g_Lhs[lhs_base + k0 + i] * (d1 * float((g_W[qbyte0 + i] >> 4) & 0xFu) - m1);
                }
            }
        }
    }
    g_Out[route * (uint)n + col] = acc;
}
